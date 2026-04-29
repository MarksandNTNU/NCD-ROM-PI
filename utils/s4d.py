"""Standalone S4D implementation.

Adapted from the official state-spaces/s4 repository:
    https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
    Gu et al., "On the Parameterization and Initialization of Diagonal
    State Space Models", NeurIPS 2022. https://arxiv.org/abs/2206.11893

Changes from original:
    - No pytorch-lightning dependency
    - No CUDA / pykeops extensions (uses pure-PyTorch naive kernels)
    - Fully MPS-compatible (Apple Silicon)
    - No einops dependency
    - S4DStack wrapper matches the LSSLStack / train_model / predict interface

Public API:
    S4DKernel  – diagonal SSM convolution kernel
    S4DLayer   – single S4D layer (FFT conv + skip + GLU output)
    S4DStack   – multi-layer encoder-SSM-decoder stack (drop-in for LSSLStack)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  DropoutNd  (ties mask across sequence dims, MPS-safe)
# ════════════════════════════════════════════════════════════════

class DropoutNd(nn.Module):
    """Dropout that ties the mask across sequence dimensions.

    Args:
        p: Dropout probability.
        tie: If True, use one mask shared across all spatial/time dims.
        transposed: If True, input has layout (B, d, L...) else (B, L..., d).
    """

    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return X

        if not self.transposed:
            # (B, L..., d) → (B, d, L...)
            dims = list(range(X.ndim))
            X = X.permute(dims[0], dims[-1], *dims[1:-1]).contiguous()

        # Build mask: tie across everything beyond (B, d)
        mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
        mask = (torch.rand(*mask_shape, device=X.device, dtype=X.dtype) >= self.p)
        X = X * mask / (1.0 - self.p)

        if not self.transposed:
            dims = list(range(X.ndim))
            X = X.permute(dims[0], *dims[2:], dims[1]).contiguous()
        return X


# ════════════════════════════════════════════════════════════════
#  S4DKernel  – diagonal SSM, ZOH discretization
# ════════════════════════════════════════════════════════════════

class S4DKernel(nn.Module):
    """Generate a convolution kernel from diagonal SSM parameters.

    Parameterization follows the S4D paper:
        A = -exp(log_A_real) + 1j * A_imag      (strictly stable)

    Initialization: S4D-Lin (linear imaginary spacing, real part = -0.5).
    The effective state dimension is d_state; internally we store d_state//2
    complex pairs (conjugate symmetry ⟹ factor-of-2 savings).

    Args:
        d_model (H): Number of independent SSM channels.
        d_state (N): Full state size (internally uses N//2 complex pairs).
        dt_min, dt_max: Uniform log range for step-size initialisation.
        lr: Optional per-parameter learning-rate override (sets _optim attr).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()
        H = d_model
        N = d_state // 2   # complex pairs; full state = 2*N (conjugate symmetry)

        # ── Step sizes (log-uniform in [dt_min, dt_max]) ─────────────────────
        log_dt = (
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self._reg("log_dt", log_dt, lr)

        # ── Diagonal A: strictly stable (real < 0, imag linear spacing) ─────
        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * torch.arange(N, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        self._reg("log_A_real", log_A_real, lr)
        self._reg("A_imag", A_imag, lr)

        # ── Output projection C (complex, stored as real view) ───────────────
        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

    def _reg(self, name: str, tensor: torch.Tensor, lr: float | None) -> None:
        """Register tensor as a parameter; attach optimizer metadata."""
        self.register_parameter(name, nn.Parameter(tensor))
        optim: dict = {"weight_decay": 0.0}   # SSM params should never be decayed
        if lr is not None:
            optim["lr"] = lr
        setattr(getattr(self, name), "_optim", optim)

    def forward(self, L: int) -> torch.Tensor:
        """Compute the SSM convolution kernel of length L.

        Returns: (H, L) real-valued kernel tensor.
        """
        dt = torch.exp(self.log_dt)                             # (H,)
        C  = torch.view_as_complex(self.C)                     # (H, N)
        A  = -torch.exp(self.log_A_real) + 1j * self.A_imag    # (H, N)

        # ZOH discretization: dtA = Δ * A
        dtA = A * dt.unsqueeze(-1)                             # (H, N)

        # Vandermonde expansion (naive, O(H N L) — MPS-compatible)
        l = torch.arange(L, device=A.device, dtype=torch.float32)
        K = dtA.unsqueeze(-1) * l                              # (H, N, L)

        # Scaled C for ZOH: C̃ = C * (exp(dtA) - 1) / A
        Cs = C * (torch.exp(dtA) - 1.0) / A                   # (H, N)

        # Sum over state dim; ×2 to account for conjugate pairs
        kernel = 2.0 * torch.einsum("hn, hnl -> hl", Cs, torch.exp(K)).real  # (H, L)
        return kernel


# ════════════════════════════════════════════════════════════════
#  S4DLayer  – single S4D layer (transposed-agnostic)
# ════════════════════════════════════════════════════════════════

class S4DLayer(nn.Module):
    """One S4D layer: FFT convolution → D skip → activation → GLU output.

    Internal computation is always in (B, H, L) layout regardless of
    the ``transposed`` flag; the flag only controls the input/output shape.

    Args:
        d_model (H): Feature dimension.
        d_state (N): SSM state size (passed through to S4DKernel).
        dropout: Dropout applied after activation.
        transposed: If True, expects (B, H, L) I/O; else (B, L, H).
        activation: Pointwise activation — 'gelu', 'silu', 'relu', or a
            pre-constructed nn.Module.
        dt_min, dt_max, lr: Forwarded to S4DKernel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        activation: str | nn.Module = "gelu",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()
        self.transposed = transposed

        self.kernel = S4DKernel(d_model, d_state=d_state, dt_min=dt_min, dt_max=dt_max, lr=lr)
        self.D = nn.Parameter(torch.randn(d_model))

        if isinstance(activation, nn.Module):
            self.activation = activation
        else:
            _act = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}
            self.activation = _act.get(activation.lower(), nn.GELU)()

        self.dropout = DropoutNd(dropout, transposed=True) if dropout > 0.0 else nn.Identity()

        # GLU output mix: (B, H, L) → (B, 2H, L) → (B, H, L)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),   # split along channel dim
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, H, L) if transposed else (B, L, H)
        returns same shape as input
        """
        if not self.transposed:
            u = u.transpose(-1, -2)   # → (B, H, L)

        L = u.size(-1)

        # Compute SSM kernel and FFT-convolve
        # Pad explicitly (avoids the n= argument which triggers MPS warnings)
        k = self.kernel(L=L)                                  # (H, L)
        k_padded = F.pad(k, (0, L))                           # (H, 2L)
        u_padded = F.pad(u, (0, L))                           # (B, H, 2L)
        k_f = torch.fft.rfft(k_padded)                        # (H, L+1)
        u_f = torch.fft.rfft(u_padded)                        # (B, H, L+1)
        y = torch.fft.irfft(u_f * k_f)[..., :L]              # (B, H, L)

        # D skip connection (learned scalar per channel)
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)                       # (B, H, L)

        if not self.transposed:
            y = y.transpose(-1, -2)                     # → (B, L, H)
        return y


# ════════════════════════════════════════════════════════════════
#  S4DStack  – multi-layer stack; drop-in for LSSLStack
# ════════════════════════════════════════════════════════════════

class S4DStack(nn.Module):
    """Stack of S4D layers with a linear encoder and MLP decoder.

    Interface is identical to LSSLStack and compatible with
    ``train_model`` / ``predict`` from ``utils.models_s4``.

    Input:  (B, seq_len, d_input)
    Output: (B, seq_len, d_output)

    Example::

        model = S4DStack(
            d_input=n_sensors, d_model=64, d_output=n_modes,
            n_layers=4, d_state=64, dropout=0.1,
            decoder_sizes=[128, 256], decoder_act=nn.SiLU(),
        ).to(device)

    Args:
        d_input: Number of sensor inputs per time step.
        d_model (H): Internal feature width.
        d_output: Number of POD modes (or any regression target).
        n_layers: Number of stacked S4D layers.
        d_state (N): SSM state size (default 64; effective state = 2*N complex pairs).
        dropout: Dropout in S4D layers and decoder.
        dt_min, dt_max: Step-size init range for the SSM kernels.
        lr: Optional learning-rate override for SSM parameters.
        decoder_sizes: Hidden widths of the MLP decoder, e.g. [128, 256].
            None → single linear projection d_model → d_output.
        decoder_act: Activation between decoder layers (default SiLU).
        device: Ignored (use ``.to(device)`` after construction).
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
        decoder_sizes: list[int] | None = None,
        decoder_act: nn.Module | None = None,
        device=None,          # ignored; use .to(device)
        **layer_args,
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList([
            S4DLayer(
                d_model,
                d_state=d_state,
                dropout=dropout,
                transposed=False,   # stack works in (B, L, H) layout
                dt_min=dt_min,
                dt_max=dt_max,
                lr=lr,
                **layer_args,
            )
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # ── Decoder ──────────────────────────────────────────────────────────
        if decoder_act is None:
            decoder_act = nn.SiLU()

        if decoder_sizes:
            sizes = [d_model] + list(decoder_sizes) + [d_output]
            dec: list[nn.Module] = []
            for i in range(len(sizes) - 1):
                dec.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    dec.append(nn.Dropout(dropout))
                    dec.append(type(decoder_act)())
            self.decoder = nn.ModuleList(dec)
        else:
            self.decoder = nn.ModuleList([nn.Linear(d_model, d_output)])

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_input)
        returns: (B, seq_len, d_output)
        """
        x = self.encoder(x)                     # (B, L, d_model)
        for layer, norm in zip(self.layers, self.norms):
            z = norm(x)                          # pre-norm
            z = layer(z)                         # (B, L, d_model)
            x = x + z                            # residual
        return self._decode(x)                   # (B, L, d_output)
