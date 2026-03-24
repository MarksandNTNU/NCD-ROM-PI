import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MPS-friendly S4 (paper-faithful ingredients, stock PyTorch only)
# ------------------------------------------------------------
# This implementation keeps the main ideas from the S4 paper:
#   1) HiPPO-LegS initialization
#   2) Normal-plus-low-rank (NPLR) parameterization
#   3) Stable diagonalization of the normal term
#   4) Bilinear discretization
#   5) Generating-function / resolvent computation at roots of unity
#   6) Woodbury correction for the low-rank term
#   7) FFT to recover the convolution kernel
#   8) D skip connection and pointwise channel mixing
#
# To stay friendly to Apple MPS and plain PyTorch, it deliberately avoids:
#   - custom CUDA / Cauchy kernels
#   - Triton / extension code
#
# Instead of the paper's near-linear Cauchy kernel, it uses a fully vectorized
# stock-PyTorch implementation of the same formula. That preserves the paper's
# algorithmic structure while remaining portable to CPU / CUDA / MPS.
#
# Shapes:
#   input  : (B, L, H)
#   output : (B, L, H)
# ============================================================


def _as_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, imag)


def _conj(x: torch.Tensor) -> torch.Tensor:
    return torch.conj(x)


def hippo_legs(N: int, *, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HiPPO-LegS matrix A and rank-1 correction vector p.

    Paper eq. (2):
        A[n, k] = -sqrt(2n+1) sqrt(2k+1)  if n > k
                = -(n+1)                  if n = k
                = 0                       if n < k

    Adding p p^T with p_n = sqrt(n + 1/2) turns A into
        A + p p^T = -1/2 I + S
    where S is skew-symmetric, enabling stable unitary diagonalization.
    """
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


@dataclass
class NPLRInit:
    Lambda: torch.Tensor  # (N,) complex
    P: torch.Tensor       # (N,) complex
    B: torch.Tensor       # (N,) complex


def make_nplr_legs(N: int, *, device=None, dtype=torch.float32) -> NPLRInit:
    """
    Construct the paper's NPLR initialization from HiPPO-LegS.

    We diagonalize the normal term A + p p^T = -1/2 I + S using a unitary basis.
    The transformed SSM is equivalent by conjugation, which preserves the input-
    output map while putting the state matrix into diagonal-plus-low-rank form.
    """
    # Eigendecomposition is more robust on CPU for this one-time init.
    cpu = torch.device("cpu")
    A, p = hippo_legs(N, device=cpu, dtype=dtype)
    S = A + torch.outer(p, p)  # = -1/2 I + skew-symmetric

    # Stable unitary diagonalization of the normal term.
    # S is real normal, so eigenvectors can be chosen unitary.
    Lambda, V = torch.linalg.eig(S)

    # Default continuous-time B from the original SSM basis.
    B = torch.sqrt(2.0 * torch.arange(N, dtype=dtype) + 1.0)

    # Conjugate into the diagonal basis.
    P = V.conj().transpose(-1, -2) @ p.to(torch.cfloat)
    B = V.conj().transpose(-1, -2) @ B.to(torch.cfloat)

    return NPLRInit(
        Lambda=Lambda.to(device=device),
        P=P.to(device=device),
        B=B.to(device=device),
    )


class S4Kernel(nn.Module):
    """
    Stock-PyTorch S4 kernel generator.

    Core paper ingredients used here:
      - HiPPO-LegS init
      - NPLR / DPLR form
      - PP* low-rank correction for added stability (paper Sec. 3.4 note)
      - bilinear discretization
      - generating function evaluated on roots of unity
      - Woodbury correction
      - inverse FFT to obtain the convolution kernel

    This version computes the Cauchy-like terms directly with broadcasting.
    That is slower than the custom kernel from the original implementation,
    but it is portable and MPS-friendly.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        l_max: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        bidirectional: bool = False,
        channels: int = 1,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.N = state_dim
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels

        init = make_nplr_legs(state_dim, device=device)
        Lambda = init.Lambda
        P = init.P
        B = init.B

        # One independent SSM per feature channel, as in the paper.
        self.log_dt = nn.Parameter(
            torch.empty(d_model, device=device).uniform_(math.log(dt_min), math.log(dt_max))
        )

        # Learn continuous-time parameters in the diagonal basis.
        # We keep real and imaginary parts explicitly as parameters.
        self.Lambda_re = nn.Parameter(Lambda.real[None, :].repeat(d_model, 1))
        self.Lambda_im = nn.Parameter(Lambda.imag[None, :].repeat(d_model, 1))

        self.P_re = nn.Parameter(P.real[None, :].repeat(d_model, 1))
        self.P_im = nn.Parameter(P.imag[None, :].repeat(d_model, 1))

        self.B_re = nn.Parameter(B.real[None, :].repeat(d_model, 1))
        self.B_im = nn.Parameter(B.imag[None, :].repeat(d_model, 1))

        # Learn C_tilde directly as recommended in Appendix C.4.
        C = torch.randn(channels, d_model, state_dim, device=device) / math.sqrt(state_dim)
        self.C_re = nn.Parameter(C)
        self.C_im = nn.Parameter(torch.zeros_like(C))

        self.D = nn.Parameter(torch.ones(channels, d_model, device=device))

    def _params(self) -> Tuple[torch.Tensor, ...]:
        Lambda = _as_complex(self.Lambda_re, self.Lambda_im)   # (H, N)
        P = _as_complex(self.P_re, self.P_im)                  # (H, N)
        B = _as_complex(self.B_re, self.B_im)                  # (H, N)
        C = _as_complex(self.C_re, self.C_im)                  # (C, H, N)

        # Stability trick: parameterize the real part to stay in the left half-plane.
        Lambda = torch.complex(-F.softplus(Lambda.real), Lambda.imag)
        dt = torch.exp(self.log_dt)                            # (H,)
        return Lambda, P, B, C, dt

    def _omega(self, L: int, device) -> torch.Tensor:
        k = torch.arange(L, device=device)
        return torch.exp(-2j * math.pi * k / L)

    def _k_gen_dplr(self, L: int) -> torch.Tensor:
        """
        Compute the truncated SSM generating function on the roots of unity,
        then recover the length-L convolution kernel with an inverse FFT.

        Returns:
            kernel of shape (channels, H, L), real-valued.
        """
        Lambda, P, B, C, dt = self._params()
        device = Lambda.device

        # Broadcast shapes
        #   H = hidden channels, N = state size, M = frequency bins
        H, N = Lambda.shape
        M = L
        omega = self._omega(M, device=device)                                # (M,)        # Bilinear resolvent variable from Lemma C.3/C.4:
        #   r(z) = 2/dt * (1-z)/(1+z)
        #
        # On the roots of unity, z = -1 can appear exactly when L is even.
        # The previous clamp_min() approach is invalid on complex tensors on MPS.
        # Instead, replace only near-zero denominators with eps + 0j.
        eps = torch.finfo(dt.dtype).eps
        z = omega
        denom = 1.0 + z
        denom = torch.where(
            denom.abs() < eps,
            torch.complex(torch.full_like(denom.real, eps), torch.zeros_like(denom.real)),
            denom,
        )
        g = (2.0 / dt[:, None]) * ((1.0 - z[None, :]) / denom[None, :])     # (H, M)

        # R(z; Lambda) = (g - Lambda)^-1
        R = 1.0 / (g[:, :, None] - Lambda[:, None, :])                       # (H, M, N)

        # Woodbury pieces for A = Lambda - P P*
        # The paper writes general PQ*, but the Sec. 3.4 note recommends PP*.
        RB = (R * B[:, None, :]).sum(dim=-1)                                 # (H, M)
        RP = (R * P[:, None, :]).sum(dim=-1)                                 # (H, M)
        PRB = (_conj(P)[:, None, :] * R * B[:, None, :]).sum(dim=-1)         # (H, M)
        PRP = (_conj(P)[:, None, :] * R * P[:, None, :]).sum(dim=-1)         # (H, M)

        # Channel-specific output projection C_tilde.
        CRB = (_conj(C)[:, :, None, :] * R[None, :, :, :] * B[None, :, None, :]).sum(dim=-1)   # (C, H, M)
        CRP = (_conj(C)[:, :, None, :] * R[None, :, :, :] * P[None, :, None, :]).sum(dim=-1)   # (C, H, M)

        correction = CRP * (PRB[None, :, :] / (1.0 + PRP[None, :, :]))
        K_hat = (2.0 / denom)[None, None, :] * (CRB - correction)            # (C, H, M)

        # Inverse FFT over the roots of unity recovers the length-L kernel.
        K = torch.fft.ifft(K_hat, n=L, dim=-1).real                           # (C, H, L)
        return K

    def forward(self, L: Optional[int] = None, rate: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            K: (channels, H, L)
            D: (channels, H)
        """
        if L is None:
            L = self.L

        if rate != 1.0:
            # Avoid mutating Parameters inside forward(); this is safer for autograd
            # and backend behavior on MPS.
            log_dt = self.log_dt + math.log(rate)
            Lambda, P, B, C, dt = self._params()
            old_dt = dt
            dt = torch.exp(log_dt)

            # Inline the kernel computation with the scaled dt.
            device = Lambda.device
            H, N = Lambda.shape
            M = L
            omega = self._omega(M, device=device)

            eps = torch.finfo(dt.dtype).eps
            z = omega
            denom = 1.0 + z
            denom = torch.where(
                denom.abs() < eps,
                torch.complex(torch.full_like(denom.real, eps), torch.zeros_like(denom.real)),
                denom,
            )
            g = (2.0 / dt[:, None]) * ((1.0 - z[None, :]) / denom[None, :])

            R = 1.0 / (g[:, :, None] - Lambda[:, None, :])
            RB = (R * B[:, None, :]).sum(dim=-1)
            RP = (R * P[:, None, :]).sum(dim=-1)
            PRB = (_conj(P)[:, None, :] * R * B[:, None, :]).sum(dim=-1)
            PRP = (_conj(P)[:, None, :] * R * P[:, None, :]).sum(dim=-1)
            CRB = (_conj(C)[:, :, None, :] * R[None, :, :, :] * B[None, :, None, :]).sum(dim=-1)
            CRP = (_conj(C)[:, :, None, :] * R[None, :, :, :] * P[None, :, None, :]).sum(dim=-1)
            correction = CRP * (PRB[None, :, :] / (1.0 + PRP[None, :, :]))
            K = torch.fft.ifft((2.0 / denom)[None, None, :] * (CRB - correction), n=L, dim=-1).real
        else:
            K = self._k_gen_dplr(L)

        if self.bidirectional:
            K_f = K
            K_b = torch.flip(K, dims=[-1])
            K = torch.cat([K_f, K_b], dim=0)
            D = torch.cat([self.D, self.D], dim=0)
        else:
            D = self.D
        return K, D


class S4Layer(nn.Module):
    """
    A deep-learning S4 block:
      depthwise global convolution (S4 kernel) + skip + pointwise mix + gate.

    Architecture choices mirror the paper's description of:
      - H independent SSMs
      - position-wise feature mixing
      - nonlinearity
      - residual connection
      - normalization
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        l_max: int = 1024,
        dropout: float = 0.0,
        prenorm: bool = True,
        bidirectional: bool = False,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.prenorm = prenorm
        self.bidirectional = bidirectional

        channels = 2  # GLU-style mixer
        self.kernel = S4Kernel(
            d_model=d_model,
            state_dim=state_dim,
            l_max=l_max,
            bidirectional=bidirectional,
            channels=channels,
            device=device,
        )

        conv_channels = channels * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Sequential(
            nn.Linear(conv_channels * d_model, 2 * d_model),
            nn.GLU(dim=-1),
        )

    def _fft_conv(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        u: (B, H, L)
        k: (C, H, L)
        returns: (B, C, H, L)
        """
        B, H, L = u.shape
        C = k.shape[0]
        n = 2 * L

        u_f = torch.fft.rfft(u, n=n, dim=-1)                                  # (B, H, F)
        k_f = torch.fft.rfft(k, n=n, dim=-1)                                  # (C, H, F)

        y_f = u_f[:, None, :, :] * k_f[None, :, :, :]                         # (B, C, H, F)
        y = torch.fft.irfft(y_f, n=n, dim=-1)[..., :L]                        # (B, C, H, L)
        return y

    def forward(self, x: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """
        x: (B, L, H)
        """
        residual = x
        if self.prenorm:
            x = self.norm(x)

        B, L, H = x.shape
        x_t = x.transpose(1, 2)                                                # (B, H, L)

        K, D = self.kernel(L=L, rate=rate)                                     # K: (C, H, L), D: (C, H)
        y = self._fft_conv(x_t, K)                                             # (B, C, H, L)
        y = y + x_t[:, None, :, :] * D[None, :, :, None]

        y = y.permute(0, 3, 1, 2).reshape(B, L, -1)                            # (B, L, C*H)
        y = self.output_linear(y)
        y = self.dropout(y)
        y = y + residual

        if not self.prenorm:
            y = self.norm(y)
        return y


class S4Stack(nn.Module):
    """
    Simple deep S4 model.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int,
        state_dim: int = 64,
        l_max: int = 1024,
        dropout: float = 0.0,
        prenorm: bool = True,
        bidirectional: bool = False,
        device=None,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(
            [
                S4Layer(
                    d_model=d_model,
                    state_dim=state_dim,
                    l_max=l_max,
                    dropout=dropout,
                    prenorm=prenorm,
                    bidirectional=bidirectional,
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, rate=rate)
        return self.decoder(x)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = S4Stack(
        d_input=32,
        d_model=128,
        d_output=10,
        n_layers=4,
        state_dim=64,
        l_max=1024,
        dropout=0.1,
        prenorm=True,
        bidirectional=False,
        device=device,
    ).to(device)

    x = torch.randn(8, 1024, 32, device=device)
    y = model(x)
    print(y.shape)
