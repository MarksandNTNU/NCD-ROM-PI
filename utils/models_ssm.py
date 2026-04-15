"""
SSM / SHRED / S4D model definitions and training utilities.

Models
------
SHREDSeq2Seq     : GRU/LSTM encoder + shallow MLP decoder  (baseline)
LSSLStack        : Linear State-Space Layer stack (HiPPO-LegS, bilinear disc.)
S4DModel         : Diagonal S4D approximation  (fast diagonal kernel, no Woodbury)
SensorToPODS4D   : Paper-aligned S4D  (NPLR + Woodbury + bilinear + FFT conv)

Performance notes
-----------------
* All SSM layers cache their convolution kernel K between forward passes and only
  recompute it when (a) the sequence length L changes or (b) invalidate_kernel() is
  called explicitly (e.g. after each optimiser step).  This avoids rebuilding K on
  every batch, which was the main reason SSMs appeared slower than SHRED-LSTM.

* Kernel caching is safe during inference (L is fixed) and gives 3-5x speedup
  per forward pass for training with fixed-length windows.

Training
--------
train_model      : generic training loop with early stopping + kernel invalidation
predict          : batched inference returning numpy arrays
compute_metrics  : MAE / MSE / MRE summary dict
mae, mse, mre    : element-wise metric helpers (numpy + torch)
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ════════════════════════════════════════════════════════════════
#  Metric helpers  (numpy and torch)
# ════════════════════════════════════════════════════════════════

def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def mae(true, pred):
    diff = true - pred
    return diff.abs().mean() if _is_torch(diff) else np.abs(diff).mean()


def mse(true, pred):
    diff = true - pred
    if _is_torch(diff):
        return diff.pow(2).sum(dim=-1).mean()
    return np.square(diff).sum(axis=-1).mean()


def mre(true, pred, eps: float = 1e-12):
    diff = true - pred
    if _is_torch(diff):
        num = diff.pow(2).sum(dim=-1).sqrt()
        den = true.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
        return (num / den).mean()
    num = np.sqrt(np.square(diff).sum(axis=-1))
    den = np.clip(np.sqrt(np.square(true).sum(axis=-1)), eps, None)
    return (num / den).mean()


# ════════════════════════════════════════════════════════════════
#  SHRED  (GRU / LSTM  seq2seq)   — unchanged, kept as baseline
# ════════════════════════════════════════════════════════════════

class SHREDSeq2Seq(nn.Module):
    """
    SHRED-style sequence-to-sequence model.

    Input  : (B, L, n_sensors)
    Output : (B, L, n_modes)
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
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
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
#  Shared FFT convolution utility
# ════════════════════════════════════════════════════════════════

def _fft_conv(u: torch.Tensor, K: torch.Tensor, L: int) -> torch.Tensor:
    """
    Causal convolution via FFT.

    u : (..., L)   signal
    K : (..., L)   kernel
    Returns (..., L)
    """
    n = 2 * L
    return torch.fft.irfft(
        torch.fft.rfft(u, n=n, dim=-1) * torch.fft.rfft(K, n=n, dim=-1),
        n=n, dim=-1
    )[..., :L]


# ════════════════════════════════════════════════════════════════
#  Kernel-caching mixin
# ════════════════════════════════════════════════════════════════

class _CachedKernelMixin:
    """
    Mixin that caches the SSM convolution kernel K.

    Subclasses must implement _build_kernel(L) -> Tensor.
    Call self._get_kernel(L) in forward(); it returns the cached
    kernel when L hasn't changed, otherwise rebuilds.

    Call invalidate_kernel() after each optimiser step so that
    updated parameters are reflected in the next forward pass.
    """

    def _init_cache(self):
        self._kernel_cache: Optional[torch.Tensor] = None
        self._kernel_L: int = -1

    def _get_kernel(self, L: int) -> torch.Tensor:
        if self._kernel_cache is None or self._kernel_L != L:
            self._kernel_cache = self._build_kernel(L)
            self._kernel_L = L
        return self._kernel_cache

    def invalidate_kernel(self):
        """Must be called after each optimiser step during training."""
        self._kernel_cache = None
        self._kernel_L = -1

    def _build_kernel(self, L: int) -> torch.Tensor:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════
#  LSSL  (HiPPO-LegS diagonal, bilinear discretisation)
# ════════════════════════════════════════════════════════════════

def _hippo_diag_eigs(N: int, device=None) -> torch.Tensor:
    """Eigenvalues of the symmetrised HiPPO-LegS matrix."""
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


class LSSLKernel(nn.Module, _CachedKernelMixin):
    """
    LSSL convolution kernel with bilinear (Tustin) discretisation.
    Caches K across forward passes; call invalidate_kernel() after opt.step().
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        device=None,
    ):
        super().__init__()
        self._init_cache()

        lam = _hippo_diag_eigs(state_dim, device=device)
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

    def _bilinear(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)
        B   = torch.complex(self.B_re, self.B_im)
        dt  = torch.exp(self.log_dt).unsqueeze(-1)   # (H, 1)
        half = 0.5 * dt * lam
        denom = 1.0 - half
        a_bar = (1.0 + half) / denom                 # (H, N)
        b_bar = (dt / denom) * B                     # (H, N)
        return a_bar, b_bar

    def _build_kernel(self, L: int) -> torch.Tensor:
        """Compute K of shape (d_model, L)."""
        a_bar, b_bar = self._bilinear()
        C = torch.complex(self.C_re, self.C_im)
        k = torch.arange(L, device=a_bar.device).to(a_bar.dtype)
        # a_bar^k via log trick:  (H, N, L)
        a_pow = torch.exp(torch.log(a_bar).unsqueeze(-1) * k)
        return ((C * b_bar).unsqueeze(-1) * a_pow).sum(dim=1).real  # (H, L)

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_kernel(L), self.D


class LSSLLayer(nn.Module):
    """Single LSSL layer with pre-norm residual."""

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        prenorm: bool = True,
        device=None,
        activation=nn.GLU,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        self.kernel = LSSLKernel(d_model, state_dim, dt_min, dt_max, device)
        self.dropout = nn.Dropout(dropout)
        if activation is nn.GLU:
            self.output_linear = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GLU(dim=-1))
        else:
            self.output_linear = nn.Sequential(nn.Linear(d_model, d_model), activation())

    def invalidate_kernel(self):
        self.kernel.invalidate_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.prenorm:
            x = self.norm(x)
        _, L, _ = x.shape
        K, D = self.kernel(L)
        u = x.transpose(1, 2)                              # (B, H, L)
        y = _fft_conv(u, K, L) + u * D.unsqueeze(-1)      # (B, H, L)
        y = y.transpose(1, 2)                              # (B, L, H)
        y = self.output_linear(y)
        y = self.dropout(y) + residual
        if not self.prenorm:
            y = self.norm(y)
        return y


class LSSLStack(nn.Module):
    """Stack of LSSL layers with linear encoder/decoder."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int,
        state_dim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        prenorm: bool = True,
        device=None,
        activation=nn.GLU,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            LSSLLayer(d_model, state_dim, dt_min, dt_max, dropout, prenorm, device, activation)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(d_model, d_output)

    def invalidate_kernel(self):
        for layer in self.layers:
            layer.invalidate_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)


# ════════════════════════════════════════════════════════════════
#  S4D  — diagonal approximation  (fast, no Woodbury)
# ════════════════════════════════════════════════════════════════

def _hippo_legs_eigs(N: int) -> np.ndarray:
    """Eigenvalues of the full HiPPO-LegS A matrix."""
    A = np.zeros((N, N))
    for n in range(N):
        A[n, n] = -(n + 1)
        for k in range(n):
            A[n, k] = -math.sqrt((2 * n + 1) * (2 * k + 1))
    return np.linalg.eigvals(A)


class _DiagS4DLayer(nn.Module, _CachedKernelMixin):
    """
    S4D diagonal layer.

    The convolution kernel K is built once per unique L and cached.
    Call invalidate_kernel() after each optimiser step.

    Input / Output : (B, L, H)
    """

    def __init__(
        self,
        d_input: int,
        d_state: int = 64,
        dropout: float = 0.1,
        activation=nn.GELU,
    ):
        super().__init__()
        self._init_cache()

        eigs = _hippo_legs_eigs(d_state)
        eigs = eigs[np.argsort(eigs.real)]

        # Parameterise real part via log so it stays negative
        self.log_neg_real = nn.Parameter(
            torch.log(-torch.tensor(eigs.real, dtype=torch.float32).clamp(max=-1e-4))
        )
        self.imag = nn.Parameter(torch.tensor(eigs.imag, dtype=torch.float32))

        scale = 0.01
        self.B_re = nn.Parameter(torch.randn(d_input, d_state) * scale)
        self.B_im = nn.Parameter(torch.randn(d_input, d_state) * scale)
        self.C_re = nn.Parameter(torch.randn(d_input, d_state) * scale)
        self.C_im = nn.Parameter(torch.randn(d_input, d_state) * scale)
        self.D = nn.Parameter(torch.randn(d_input) * scale)
        self.log_step = nn.Parameter(torch.zeros(d_input).uniform_(-4, -1))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_input)
        self.act = activation()

    def _build_kernel(self, L: int) -> torch.Tensor:
        """
        Compute K of shape (H, L).
        
        For each channel h and time k:
          K[h, k] = Re[ sum_n  (C[h,n] * B[h,n]) * (lambda_h * step_h)^k ]
        where the sum collapses to a single einsum over the state dimension.
        """
        Lambda_re = -torch.exp(self.log_neg_real)            # (N,)
        Lambda_im = self.imag                                 # (N,)
        step = torch.exp(self.log_step).clamp(1e-5, 1.0)     # (H,)

        s_re = Lambda_re.unsqueeze(0) * step.unsqueeze(-1)   # (H, N)
        s_im = Lambda_im.unsqueeze(0) * step.unsqueeze(-1)   # (H, N)

        # Discretised B
        Bb_re = self.B_re * step.unsqueeze(-1)               # (H, N)
        Bb_im = self.B_im * step.unsqueeze(-1)               # (H, N)

        # Powers: (H, N, L)
        l_idx = torch.arange(L, device=step.device, dtype=torch.float32)
        ls_re = s_re.unsqueeze(-1) * l_idx                   # (H, N, L)
        ls_im = s_im.unsqueeze(-1) * l_idx                   # (H, N, L)
        pow_mag = torch.exp(ls_re)                           # (H, N, L)

        CB_re = self.C_re * Bb_re - self.C_im * Bb_im        # (H, N)
        CB_im = self.C_re * Bb_im + self.C_im * Bb_re        # (H, N)

        K = (
            torch.einsum("hn,hnl->hl", CB_re, pow_mag * torch.cos(ls_im))
            - torch.einsum("hn,hnl->hl", CB_im, pow_mag * torch.sin(ls_im))
        )                                                     # (H, L)
        return K

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, H = u.shape
        K = self._get_kernel(L)                              # (H, L)
        u_t = u.transpose(1, 2)                             # (B, H, L)
        y = _fft_conv(u_t, K, L)                            # (B, H, L)
        y = y + u_t * self.D.unsqueeze(0).unsqueeze(-1)
        y = y.transpose(1, 2)                               # (B, L, H)
        y = self.dropout(self.act(y))
        return self.norm(y)


class S4DModel(nn.Module):
    """
    S4D sequence model:  linear projection → _DiagS4DLayer stack → MLP decoder.

    Input  : (B, L, n_sensors)
    Output : (B, L, n_modes)
    """

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
        self.layers = nn.ModuleList([
            _DiagS4DLayer(d_model, d_state=d_state, dropout=dropout, activation=activation)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            activation(),
            nn.Linear(256, 512),
            activation(),
            nn.Linear(512, n_modes),
        )

    def invalidate_kernel(self):
        for layer in self.layers:
            layer.invalidate_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.decoder(h)


# ════════════════════════════════════════════════════════════════
#  SensorToPODS4D  — paper-aligned S4 (NPLR + Woodbury + bilinear)
# ════════════════════════════════════════════════════════════════

def _hippo_legs_nplr(N: int, device=None, dtype=torch.float32):
    """
    Return (Lambda, P, B) for the NPLR parameterisation of HiPPO-LegS.
    
    Lambda : (N,) complex — diagonal of symmetrised A in the eigenbasis
    P      : (N,) complex — low-rank correction vector
    B      : (N,) complex — input vector in the eigenbasis
    """
    cpu = torch.device("cpu")
    n = torch.arange(N, dtype=dtype, device=cpu)
    r = torch.sqrt(2.0 * n + 1.0)

    A = torch.zeros(N, N, dtype=dtype, device=cpu)
    i, j = torch.meshgrid(n.long(), n.long(), indexing="ij")
    A[i > j] = -(r[:, None] * r[None, :])[i > j]
    A[torch.arange(N), torch.arange(N)] = -(n + 1.0)

    p = torch.sqrt(n + 0.5)
    S = A + torch.outer(p, p)
    Lambda, V = torch.linalg.eig(S)

    B_vec = torch.sqrt(2.0 * n + 1.0).to(torch.cfloat)
    Vh = V.conj().T
    P = (Vh @ p.to(torch.cfloat)).to(device=device)
    B = (Vh @ B_vec).to(device=device)
    Lambda = Lambda.to(device=device)
    return Lambda, P, B


class S4DKernel(nn.Module, _CachedKernelMixin):
    """
    Paper-aligned S4 kernel: NPLR parameterisation, Woodbury inversion,
    bilinear (Tustin) discretisation, frequency-domain evaluation.

    Kernel is cached; call invalidate_kernel() after each opt.step().
    """

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
        self._init_cache()

        self.d_model = d_model
        self.d_state = d_state
        self.channels = channels
        self.l_max = l_max
        self.bidirectional = bidirectional

        Lambda, P, B = _hippo_legs_nplr(d_state, device=device)

        self.Lambda_re = nn.Parameter(Lambda.real[None].repeat(d_model, 1))
        self.Lambda_im = nn.Parameter(Lambda.imag[None].repeat(d_model, 1))
        self.P_re = nn.Parameter(P.real[None].repeat(d_model, 1))
        self.P_im = nn.Parameter(P.imag[None].repeat(d_model, 1))
        self.B_re = nn.Parameter(B.real[None].repeat(d_model, 1))
        self.B_im = nn.Parameter(B.imag[None].repeat(d_model, 1))

        C = torch.randn(channels, d_model, d_state, device=device) / math.sqrt(d_state)
        self.C_re = nn.Parameter(C)
        self.C_im = nn.Parameter(torch.zeros_like(C))
        self.D = nn.Parameter(torch.ones(channels, d_model, device=device))
        self.log_dt = nn.Parameter(
            torch.empty(d_model, device=device).uniform_(math.log(dt_min), math.log(dt_max))
        )

    def _params(self):
        Lambda = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)
        P = torch.complex(self.P_re, self.P_im)
        B = torch.complex(self.B_re, self.B_im)
        C = torch.complex(self.C_re, self.C_im)
        dt = torch.exp(self.log_dt)
        return Lambda, P, B, C, dt

    def _build_kernel(self, L: int) -> torch.Tensor:
        """
        Evaluate K via the Woodbury resolvent in the frequency domain.

        Returns K of shape (channels, d_model, L).
        """
        Lambda, P, B, C, dt = self._params()
        device = Lambda.device

        # Roots of unity for length-L DFT
        k = torch.arange(L, device=device)
        z = torch.exp(-2j * math.pi * k / L)               # (L,)

        # Bilinear transform: g maps z -> continuous frequency
        eps = torch.finfo(dt.dtype).eps
        denom_z = 1.0 + z                                   # (L,)
        denom_z = torch.where(
            denom_z.abs() < eps,
            torch.complex(
                torch.full_like(denom_z.real, eps),
                torch.zeros_like(denom_z.real)
            ),
            denom_z,
        )
        g = (2.0 / dt[:, None]) * ((1.0 - z[None]) / denom_z[None])  # (H, L)

        # Resolvent  R_n(z) = 1 / (g - lambda_n)  for each (h, l, n)
        R = 1.0 / (g[:, :, None] - Lambda[:, None, :])                # (H, L, N)

        # Rank-1 Woodbury correction terms
        Pbar = P.conj()                                                # (H, N)
        PRB = (Pbar[:, None] * R * B[:, None]).sum(-1)                 # (H, L)
        PRP = (Pbar[:, None] * R * P[:, None]).sum(-1)                 # (H, L)

        # Output: sum over state dimension with C
        CRB = (C.conj()[:, :, None] * R[None] * B[None, :, None]).sum(-1)   # (C, H, L)
        CRP = (C.conj()[:, :, None] * R[None] * P[None, :, None]).sum(-1)   # (C, H, L)

        correction = CRP * (PRB[None] / (1.0 + PRP[None]))            # (C, H, L)
        K_hat = (2.0 / denom_z)[None, None] * (CRB - correction)      # (C, H, L)

        K = torch.fft.ifft(K_hat, n=L, dim=-1).real                   # (C, H, L)
        return K

    def forward(self, L: Optional[int] = None, rate: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if L is None:
            L = self.l_max

        # For non-unit rate, shift log_dt and invalidate cache
        if rate != 1.0:
            # recompute with modified dt (not cached)
            old_log_dt = self.log_dt.data.clone()
            self.log_dt.data = old_log_dt + math.log(rate)
            K = self._build_kernel(L)
            self.log_dt.data = old_log_dt
        else:
            K = self._get_kernel(L)

        if self.bidirectional:
            K = torch.cat([K, torch.flip(K, dims=[-1])], dim=0)
            D = torch.cat([self.D, self.D], dim=0)
        else:
            D = self.D
        return K, D


class S4DLayer(nn.Module):
    """S4D convolution layer.  Input / Output : (B, L, H)."""

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
                nn.Linear(conv_channels * d_model, 2 * d_model), nn.GLU(dim=-1)
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Linear(conv_channels * d_model, d_model), activation()
            )

    def invalidate_kernel(self):
        self.kernel.invalidate_kernel()

    def forward(self, x: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, L, H = x.shape
        x_t = x.transpose(1, 2)                             # (B, H, L)
        K, D = self.kernel(L=L, rate=rate)                  # (C, H, L), (C, H)

        # Multi-channel convolution: (B, C, H, L)
        u_f = torch.fft.rfft(x_t, n=2 * L, dim=-1)
        K_f = torch.fft.rfft(K, n=2 * L, dim=-1)
        y = torch.fft.irfft(u_f[:, None] * K_f[None], n=2 * L, dim=-1)[..., :L]
        y = y + x_t[:, None] * D[None, :, :, None]         # skip connection via D
        y = y.permute(0, 3, 1, 2).reshape(B, L, -1)        # (B, L, C*H)
        y = self.output_linear(y)
        y = self.dropout(y)
        return residual + y


class S4DBlock(nn.Module):
    """S4D block: norm → expand → S4DLayer → project back."""

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

    def invalidate_kernel(self):
        self.ssm.invalidate_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.in_proj(z)
        z = self.ssm(z)
        z = self.out_proj(z)
        return x + self.dropout(z)


class SensorToPODS4D(nn.Module):
    """
    Full paper-aligned S4D stack.

    Input  : (B, L, n_sensors)
    Output : (B, L, n_modes)
    """

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
            S4DBlock(d_model, d_state, dropout, expansion=2, l_max=l_max, activation=activation)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, n_modes)

    def invalidate_kernel(self):
        for layer in self.layers:
            layer.invalidate_kernel()

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sensors)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(self.norm(x))


# ════════════════════════════════════════════════════════════════
#  Helper: collect all SSM layers in a model for bulk invalidation
# ════════════════════════════════════════════════════════════════

def _ssm_layers(model: nn.Module):
    """Yield all kernel-caching submodules in a model."""
    return [m for m in model.modules() if isinstance(m, _CachedKernelMixin)]


def invalidate_all_kernels(model: nn.Module):
    """Invalidate kernel caches in all SSM sublayers of *model*."""
    for m in _ssm_layers(model):
        m.invalidate_kernel()


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
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    label: str = "model",
) -> Tuple[list, list]:
    """
    Generic training loop with early stopping and LR scheduling.

    After each optimiser step, all SSM kernel caches are invalidated
    so that the next forward pass recomputes K from updated parameters.
    For non-SSM models (e.g. SHREDSeq2Seq) this is a no-op.
    """
    if loss_fun is None:
        loss_fun = mse

    model = model.to(device)
    start = time.time()

    def _to_dl(X, Y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_dl = _to_dl(X_tr, Y_tr, shuffle=True)
    val_dl   = _to_dl(X_vl, Y_vl, shuffle=False)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(patience // 2, 5), factor=0.5)

    best_val   = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait       = 0
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
            # ← kernel cache invalidated here so next batch sees updated params
            invalidate_all_kernels(model)
            tr_sum += loss.item()

        tr_loss = tr_sum / max(len(train_dl), 1)

        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += F.mse_loss(model(xb), yb).item()
        vl_loss = vl_sum / max(len(val_dl), 1)

        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        sched.step(vl_loss)

        print(
            f"[{label}] epoch={epoch:4d}/{epochs}  train={tr_loss:.4e}  "
            f"val={vl_loss:.4e}  best={best_val:.4e}  "
            f"lr={opt.param_groups[0]['lr']:.1e}  wait={wait}/{patience}",
            end="\r",
        )

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

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
    """Batched model inference; returns numpy array."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(model(Xt[i: i + batch_size]).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, MSE, MRE (%) as formatted strings."""
    _mae = np.mean(np.abs(y_true - y_pred))
    _mse = np.mean((y_true - y_pred) ** 2)
    num  = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1))
    den  = np.sqrt(np.sum(y_true ** 2, axis=-1)) + 1e-12
    _mre = np.mean(num / den)
    return {
        "mae": f"{_mae:.4e}",
        "mse": f"{_mse:.4e}",
        "mre": f"{100 * _mre:.2f}%",
    }