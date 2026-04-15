"""
SSM / SHRED / LSSL / S4D model definitions and training utilities.

Models:
    - SHREDSeq2Seq   : GRU/LSTM encoder + shallow MLP decoder
    - LSSLStack       : Linear State-Space Layer stack (HiPPO-LegS, bilinear)
    - SensorToPODS4D  : Paper-aligned S4D (NPLR + Woodbury + FFT convolution)

Training:
    - train_model     : generic training loop with early stopping
    - predict         : batched inference returning numpy arrays
    - compute_metrics : MAE / MSE / MRE summary dict (numpy)
    - mae, mse, mre   : element-wise metric functions (numpy + torch)
"""
from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ════════════════════════════════════════════════════════════════
#  Metric helpers (work with both torch tensors and numpy arrays)
# ════════════════════════════════════════════════════════════════

def _is_torch(x):
    return isinstance(x, torch.Tensor)


def mae(datatrue, datapred):
    diff = datatrue - datapred
    if _is_torch(diff):
        return diff.abs().mean()
    return np.abs(diff).mean()


def mse(datatrue, datapred):
    diff = datatrue - datapred
    if _is_torch(diff):
        return diff.pow(2).sum(dim=-1).mean()
    return np.square(diff).sum(axis=-1).mean()


def mre(datatrue, datapred, eps=1e-12):
    diff = datatrue - datapred
    if _is_torch(diff):
        num = diff.pow(2).sum(dim=-1).sqrt()
        den = datatrue.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
        return (num / den).mean()
    num = np.sqrt(np.square(diff).sum(axis=-1))
    den = np.sqrt(np.square(datatrue).sum(axis=-1))
    den = np.clip(den, eps, None)
    return (num / den).mean()




# ════════════════════════════════════════════════════════════════
#  SHRED  (GRU / LSTM  seq2seq)
# ════════════════════════════════════════════════════════════════

class SHREDSeq2Seq(nn.Module):
    """
    SHRED-style sequence-to-sequence model:
    sparse sensor sequence -> recurrent latent -> shallow decoder -> output sequence.

    Input:  (B, L, n_sensors)
    Output: (B, L, n_modes)
    """

    def __init__(
        self,
        n_sensors: int,
        n_modes: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
        activation=nn.GELU,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        rnn_dropout = dropout if n_layers > 1 else 0.0
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=n_sensors,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=n_sensors,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        d_rnn = hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.Sequential(
            nn.Linear(d_rnn, decoder_hidden // 2),
            activation(),
            nn.Linear(decoder_hidden // 2, decoder_hidden),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, n_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.decoder(h)


# ════════════════════════════════════════════════════════════════
#  LSSL  (Linear State-Space Layer — HiPPO-LegS, bilinear disc.)
# ════════════════════════════════════════════════════════════════

def _lssl_hippo_diag(N: int, device=None):
    cpu = torch.device("cpu")
    n = torch.arange(N, dtype=torch.float32, device=cpu)
    r = torch.sqrt(2.0 * n + 1.0)
    A = torch.zeros(N, N, dtype=torch.float32, device=cpu)
    ii, jj = torch.meshgrid(n.long(), n.long(), indexing="ij")
    A[ii > jj] = -(r[ii[ii > jj]] * r[jj[ii > jj]])
    A.diagonal().copy_(-(n + 1.0))
    p = torch.sqrt(n + 0.5)
    S = A + torch.outer(p, p)
    lam, _ = torch.linalg.eig(S)
    return lam[lam.imag.argsort()].to(device)


class LSSLKernel(nn.Module):
    def __init__(self, d_model, state_dim, l_max, dt_min=1e-3, dt_max=1e-1, device=None):
        super().__init__()
        lam = _lssl_hippo_diag(state_dim, device=device)
        self.Lambda_re = nn.Parameter(lam.real.unsqueeze(0).repeat(d_model, 1))
        self.Lambda_im = nn.Parameter(lam.imag.unsqueeze(0).repeat(d_model, 1))
        scale = 1.0 / math.sqrt(state_dim)
        self.B_re = nn.Parameter(torch.randn(d_model, state_dim, device=device) * scale)
        self.B_im = nn.Parameter(torch.zeros(d_model, state_dim, device=device))
        self.C_re = nn.Parameter(torch.randn(d_model, state_dim, device=device) * scale)
        self.C_im = nn.Parameter(torch.zeros(d_model, state_dim, device=device))
        self.D = nn.Parameter(torch.ones(d_model, device=device))
        self.log_dt = nn.Parameter(
            torch.empty(d_model, device=device).uniform_(math.log(dt_min), math.log(dt_max))
        )

    def _bilinear(self):
        lam = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)
        B = torch.complex(self.B_re, self.B_im)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        half = 0.5 * dt * lam
        denom = 1.0 - half
        a_bar = (1.0 + half) / denom
        b_bar = (dt / denom) * B
        return a_bar, b_bar

    def kernel(self, L):
        a_bar, b_bar = self._bilinear()
        C = torch.complex(self.C_re, self.C_im)
        k = torch.arange(L, device=a_bar.device)
        a_pow = torch.exp(torch.log(a_bar).unsqueeze(-1) * k.to(a_bar.dtype))
        return ((C * b_bar).unsqueeze(-1) * a_pow).sum(dim=1).real

    def forward(self, L):
        return self.kernel(L), self.D


class LSSLLayer(nn.Module):
    def __init__(
        self,
        d_model,
        state_dim=64,
        l_max=2048,
        dt_min=1e-3,
        dt_max=1e-1,
        dropout=0.0,
        prenorm=True,
        device=None,
        activation=nn.GLU,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        self.kernel = LSSLKernel(d_model, state_dim, l_max, dt_min, dt_max, device)
        self.dropout = nn.Dropout(dropout)
        if activation is nn.GLU:
            self.output_linear = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GLU(dim=-1),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Linear(d_model, d_model),
                activation(),
            )

    def _fft_conv(self, u, k):
        L = u.shape[-1]
        n = 2 * L
        uf = torch.fft.rfft(u, n=n, dim=-1)
        kf = torch.fft.rfft(k, n=n, dim=-1)
        return torch.fft.irfft(uf * kf.unsqueeze(0), n=n, dim=-1)[..., :L]

    def forward(self, x):
        residual = x
        if self.prenorm:
            x = self.norm(x)
        _, L, _ = x.shape
        K, D = self.kernel(L)
        u = x.transpose(1, 2)
        y = self._fft_conv(u, K) + u * D.unsqueeze(-1)
        y = y.transpose(1, 2)
        y = self.output_linear(y)
        y = self.dropout(y)
        y = y + residual
        if not self.prenorm:
            y = self.norm(y)
        return y


class LSSLStack(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        d_output,
        n_layers,
        state_dim=64,
        l_max=2048,
        dt_min=1e-3,
        dt_max=1e-1,
        dropout=0.0,
        prenorm=True,
        device=None,
        activation=nn.GLU,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            LSSLLayer(d_model, state_dim, l_max, dt_min, dt_max, dropout, prenorm, device, activation)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)


# ════════════════════════════════════════════════════════════════
#  S4D  (Paper-aligned: NPLR + Woodbury + bilinear + FFT)
# ════════════════════════════════════════════════════════════════

def _as_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.complex(re, im)


def _conj(x: torch.Tensor) -> torch.Tensor:
    return torch.conj(x)


def hippo_legs_matrix(N: int, device=None, dtype=torch.float32):
    """HiPPO-LegS matrix A and rank-1 correction vector p."""
    n = torch.arange(N, device=device, dtype=dtype)
    r = torch.sqrt(2.0 * n + 1.0)

    A = torch.zeros(N, N, device=device, dtype=dtype)
    i = n[:, None]
    j = n[None, :]

    lower = i > j
    A[lower] = -(r[:, None] * r[None, :])[lower]
    A[torch.arange(N), torch.arange(N)] = -(n + 1.0)

    p = torch.sqrt(n + 0.5)
    return A, p


def make_nplr_legs(N: int, device=None, dtype=torch.float32):
    """NPLR parameterization from HiPPO-LegS (diagonalisation at init)."""
    cpu = torch.device("cpu")
    A, p = hippo_legs_matrix(N, device=cpu, dtype=dtype)

    S = A + torch.outer(p, p)
    Lambda, V = torch.linalg.eig(S)

    B = torch.sqrt(2.0 * torch.arange(N, dtype=dtype) + 1.0)
    Vh = V.conj().transpose(-1, -2)
    P = Vh @ p.to(torch.cfloat)
    B = Vh @ B.to(torch.cfloat)

    return (
        Lambda.to(device=device),
        P.to(device=device),
        B.to(device=device),
    )


class S4DKernel(nn.Module):
    """Paper-aligned S4 kernel with Woodbury correction and bilinear discretization."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        channels: int = 1,
        l_max: int = 1024,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        bidirectional: bool = False,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.channels = channels
        self.l_max = l_max
        self.bidirectional = bidirectional

        Lambda, P, B = make_nplr_legs(d_state, device=device)

        self.Lambda_re = nn.Parameter(Lambda.real[None, :].repeat(d_model, 1))
        self.Lambda_im = nn.Parameter(Lambda.imag[None, :].repeat(d_model, 1))
        self.P_re = nn.Parameter(P.real[None, :].repeat(d_model, 1))
        self.P_im = nn.Parameter(P.imag[None, :].repeat(d_model, 1))
        self.B_re = nn.Parameter(B.real[None, :].repeat(d_model, 1))
        self.B_im = nn.Parameter(B.imag[None, :].repeat(d_model, 1))

        C = torch.randn(channels, d_model, d_state, device=device) / math.sqrt(d_state)
        self.C_re = nn.Parameter(C)
        self.C_im = nn.Parameter(torch.zeros_like(C))
        self.D = nn.Parameter(torch.ones(channels, d_model, device=device))
        self.log_dt = nn.Parameter(
            torch.empty(d_model, device=device).uniform_(math.log(dt_min), math.log(dt_max))
        )

    def _params(self):
        Lambda = _as_complex(self.Lambda_re, self.Lambda_im)
        P = _as_complex(self.P_re, self.P_im)
        B = _as_complex(self.B_re, self.B_im)
        C = _as_complex(self.C_re, self.C_im)
        Lambda = torch.complex(-F.softplus(Lambda.real), Lambda.imag)
        dt = torch.exp(self.log_dt)
        return Lambda, P, B, C, dt

    def _omega(self, L: int, device):
        k = torch.arange(L, device=device)
        return torch.exp(-2j * math.pi * k / L)

    def _compute_kernel_with_dt(self, L: int, dt: torch.Tensor):
        Lambda, P, B, C, _ = self._params()
        device = Lambda.device
        z = self._omega(L, device=device)

        eps = torch.finfo(dt.dtype).eps
        denom = 1.0 + z
        denom = torch.where(
            denom.abs() < eps,
            torch.complex(torch.full_like(denom.real, eps), torch.zeros_like(denom.real)),
            denom,
        )

        g = (2.0 / dt[:, None]) * ((1.0 - z[None, :]) / denom[None, :])
        R = 1.0 / (g[:, :, None] - Lambda[:, None, :])

        PRB = (_conj(P)[:, None, :] * R * B[:, None, :]).sum(dim=-1)
        PRP = (_conj(P)[:, None, :] * R * P[:, None, :]).sum(dim=-1)

        CRB = (_conj(C)[:, :, None, :] * R[None, :, :, :] * B[None, :, None, :]).sum(dim=-1)
        CRP = (_conj(C)[:, :, None, :] * R[None, :, :, :] * P[None, :, None, :]).sum(dim=-1)

        correction = CRP * (PRB[None, :, :] / (1.0 + PRP[None, :, :]))
        K_hat = (2.0 / denom)[None, None, :] * (CRB - correction)
        return torch.fft.ifft(K_hat, n=L, dim=-1).real

    def forward(self, L=None, rate: float = 1.0):
        if L is None:
            L = self.l_max
        dt = torch.exp(self.log_dt + math.log(rate))
        K = self._compute_kernel_with_dt(L, dt)

        if self.bidirectional:
            K = torch.cat([K, torch.flip(K, dims=[-1])], dim=0)
            D = torch.cat([self.D, self.D], dim=0)
        else:
            D = self.D
        return K, D


class S4DLayer(nn.Module):
    """S4D convolution layer.  Input/Output: (B, L, H)."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        l_max: int = 1024,
        activation=nn.GLU,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(
            d_model=d_model,
            d_state=d_state,
            channels=2,
            l_max=l_max,
            bidirectional=bidirectional,
        )
        conv_channels = 2 * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation is nn.GLU:
            self.output_linear = nn.Sequential(
                nn.Linear(conv_channels * d_model, 2 * d_model),
                nn.GLU(dim=-1),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Linear(conv_channels * d_model, d_model),
                activation(),
            )

    def _fft_conv(self, u, k):
        B, H, L = u.shape
        n = 2 * L
        u_f = torch.fft.rfft(u, n=n, dim=-1)
        k_f = torch.fft.rfft(k, n=n, dim=-1)
        y_f = u_f[:, None, :, :] * k_f[None, :, :, :]
        return torch.fft.irfft(y_f, n=n, dim=-1)[..., :L]

    def forward(self, x: torch.Tensor, rate: float = 1.0):
        residual = x
        x = self.norm(x)
        B, L, H = x.shape
        x_t = x.transpose(1, 2)
        K, D = self.kernel(L=L, rate=rate)
        y = self._fft_conv(x_t, K)
        y = y + x_t[:, None, :, :] * D[None, :, :, None]
        y = y.permute(0, 3, 1, 2).reshape(B, L, -1)
        y = self.output_linear(y)
        y = self.dropout(y)
        return residual + y


class S4DBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        expansion: int = 2,
        l_max: int = 1024,
        activation=nn.GLU,
    ):
        super().__init__()
        inner = expansion * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner)
        self.ssm = S4DLayer(
            d_model=inner,
            d_state=d_state,
            dropout=dropout,
            bidirectional=False,
            l_max=l_max,
            activation=activation,
        )
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        z = self.norm(x)
        z = self.in_proj(z)
        z = self.ssm(z)
        z = self.out_proj(z)
        z = self.dropout(z)
        return x + z


class SensorToPODS4D(nn.Module):
    """Full S4D stack: (B, L, n_sensors) -> (B, L, n_modes)."""

    def __init__(
        self,
        n_sensors: int,
        n_modes: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        l_max: int = 1024,
        activation=nn.GLU,
    ):
        super().__init__()
        self.encoder = nn.Linear(n_sensors, d_model)
        self.layers = nn.ModuleList([
            S4DBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                expansion=2,
                l_max=l_max,
                activation=activation,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, n_modes)

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sensors)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.decoder(x)


# ════════════════════════════════════════════════════════════════
#  S4D — diagonal approximation (HiPPO eigenvalues + FFT conv)
# ════════════════════════════════════════════════════════════════

def _make_hippo_legs(N: int):
    """Return HiPPO-LegS matrices (A, B) as numpy arrays."""
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    for n in range(N):
        B[n, 0] = (2 * n + 1) ** 0.5
        for k in range(N):
            if n > k:
                A[n, k] = -(2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
            elif n == k:
                A[n, k] = -(n + 1)
    return A, B


class _DiagS4DLayer(nn.Module):
    """S4D diagonal approximation with per-channel step sizes, MPS-safe real arithmetic."""

    def __init__(self, d_input: int, d_state: int = 64, dropout: float = 0.1, activation=nn.GELU):
        super().__init__()
        A_np, _ = _make_hippo_legs(d_state)
        eigs = np.linalg.eigvals(A_np)
        idx = np.argsort(eigs.real)
        Lambda = eigs[idx]

        self.log_neg_real = nn.Parameter(
            torch.log(-torch.tensor(Lambda.real, dtype=torch.float32).clamp(max=-1e-4))
        )
        self.imag = nn.Parameter(torch.tensor(Lambda.imag, dtype=torch.float32))
        self.B_re = nn.Parameter(torch.randn(d_input, d_state) * 0.01)
        self.B_im = nn.Parameter(torch.randn(d_input, d_state) * 0.01)
        self.C_re = nn.Parameter(torch.randn(d_input, d_state) * 0.01)
        self.C_im = nn.Parameter(torch.randn(d_input, d_state) * 0.01)
        self.D = nn.Parameter(torch.randn(d_input) * 0.01)
        self.log_step = nn.Parameter(torch.zeros(d_input).uniform_(-4, -1))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_input)
        self.act = activation()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, H = u.shape
        Lambda_re = -torch.exp(self.log_neg_real)
        Lambda_im = self.imag
        step = torch.exp(self.log_step).clamp(min=1e-5, max=1.0)

        s_re = Lambda_re.unsqueeze(0) * step.unsqueeze(-1)
        s_im = Lambda_im.unsqueeze(0) * step.unsqueeze(-1)

        Bb_re = self.B_re * step.unsqueeze(-1)
        Bb_im = self.B_im * step.unsqueeze(-1)

        l_idx = torch.arange(L, device=u.device, dtype=torch.float32)
        ls_re = s_re.unsqueeze(1) * l_idx.unsqueeze(0).unsqueeze(-1)
        ls_im = s_im.unsqueeze(1) * l_idx.unsqueeze(0).unsqueeze(-1)
        pow_mag = torch.exp(ls_re)
        pow_cos = torch.cos(ls_im)
        pow_sin = torch.sin(ls_im)

        CB_re = self.C_re * Bb_re - self.C_im * Bb_im
        CB_im = self.C_re * Bb_im + self.C_im * Bb_re

        K = (
            torch.einsum("hn,hln->hl", CB_re, pow_mag * pow_cos)
            - torch.einsum("hn,hln->hl", CB_im, pow_mag * pow_sin)
        )

        u_f = torch.fft.rfft(u.transpose(1, 2), n=2 * L)
        K_f = torch.fft.rfft(K, n=2 * L)
        y = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2 * L)[..., :L]
        y = y + u.transpose(1, 2) * self.D.unsqueeze(0).unsqueeze(-1)
        y = y.transpose(1, 2)
        y = self.dropout(self.act(y))
        y = self.norm(y)
        return y


class S4DModel(nn.Module):
    """Generic S4D sequence model: Linear proj -> _DiagS4DLayer stack -> MLP decoder."""

    def __init__(
        self,
        n_sensors: int,
        n_modes: int,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        activation=nn.GELU,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, d_model)
        self.layers = nn.ModuleList(
            [_DiagS4DLayer(d_model, d_state=d_state, dropout=dropout, activation=activation) for _ in range(n_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            activation(),
            nn.Linear(256, 512),
            activation(),
            nn.Linear(512, n_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.decoder(h)


# ════════════════════════════════════════════════════════════════
#  Training / inference utilities
# ════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_vl: np.ndarray,
    Y_vl: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    loss_fun=None,
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 20,
    lr_patience: int = 10,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    label: str = "model",
):
    """Generic training loop with early stopping and LR scheduling."""
    if loss_fun is None:
        loss_fun = mse

    model = model.to(device)
    start = time.time()

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_vl, dtype=torch.float32),
            torch.tensor(Y_vl, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=lr_patience, factor=0.5)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
    tr_losses, vl_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fun(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_sum += loss.item()

        tr_loss = tr_sum / max(len(train_dl), 1)

        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += loss_fun(model(xb), yb).item()
        vl_loss = vl_sum / max(len(val_dl), 1)

        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        sched.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(
            f"[{label}] epoch={epoch:4d}/{epochs}  train={tr_loss:.4e}  "
            f"val={vl_loss:.4e}  best={best_val:.4e}  "
            f"lr={opt.param_groups[0]['lr']:.1e}  wait={wait}/{patience}",
            end="\r",
        )

        if wait >= patience:
            print(f"\n[{label}] Early stop at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\n[{label}] Training completed in {elapsed:.2f}s")
    model.load_state_dict(best_state)
    model.to(device)
    return tr_losses, vl_losses


def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched model inference, returns numpy array."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(model(Xt[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, MSE, MRE (%) as formatted strings."""
    _mae = np.mean(np.abs(y_true - y_pred))
    _mse = np.mean((y_true - y_pred) ** 2)
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1))
    den = np.sqrt(np.sum(y_true ** 2, axis=-1)) + 1e-12
    _mre = np.mean(num / den)
    return {"mae": f"{_mae:.4e}", "mse": f"{_mse:.4e}", "mre": f"{100 * _mre:.2f}%"}
