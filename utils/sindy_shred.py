"""
SINDy-SHRED: Sparse Identification of Nonlinear Dynamics + Shallow Recurrent Decoder

Fully self-contained in PyTorch (MPS / CUDA / CPU) + NumPy.
No JAX dependency.

Public API
----------
# SINDy (numpy)
build_sindy_library(A, poly_order, include_bias)   -> (Theta, feature_names)
discover_sindy_dynamics(A, dA, ...)                -> (Xi, info_dict)
sindy_integrate(Xi, a0, t, poly_order)             -> A_pred  (RK4)
time_derivative(A, dt, order=4)                    -> dA  (4th-order FD on time axis)

# SHRED (PyTorch)
SHRED(n_sensors, n_modes, ...)                     -> nn.Module
train_shred(model, X_tr, Y_tr, X_val, Y_val, ...) -> (train_losses, val_losses)
train_sindy_shred(model, ..., X_tr_traj, Xi,       -> (train_losses, val_losses, Xi_final)
                  scaler, dt, ...)  [joint SINDy reg on POD modes]
shred_predict(model, X, ...)                       -> Y_pred (numpy)

# Metrics
compute_metrics(y_true, y_pred)                    -> dict(mae, mse, mre, max_err)
evaluate_reconstruction(modes_pred, modes_true,    -> dict(...)
                        Phi, snaps_true)

# Combined
SINDySHRED                                          -> wrapper class
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from copy import deepcopy
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Device helper
# ============================================================================

def get_device() -> torch.device:
    """Return MPS → CUDA → CPU in priority order."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Metric lambdas (PyTorch tensors)
# ============================================================================
#
# MSE  : sum squared residual over last axis (modes / spatial), mean over samples
# MRE  : per-sample relative L2 error, then mean
# MAE  : element-wise mean absolute error
# num2p: float probability → "XX.XX%" string

mae_fn  = lambda yt, yp: (yt - yp).abs().mean()
mse_fn  = lambda yt, yp: (yt - yp).pow(2).sum(dim=-1).mean()
mre_fn  = lambda yt, yp: (
    (yt - yp).pow(2).sum(dim=-1).sqrt() /
    (yt.pow(2).sum(dim=-1).sqrt() + 1e-12)
).mean()
num2p   = lambda prob: f"{100 * float(prob):.2f}%"


# ============================================================================
# SINDy — pure NumPy
# ============================================================================

def time_derivative(
    A: np.ndarray,
    dt: float,
    order: int = 4,
) -> np.ndarray:
    """
    Compute dA/dt along axis-0 (time axis) using finite differences.

    Parameters
    ----------
    A     : (N, n_modes)  — one trajectory of POD modal coefficients
    dt    : uniform time step
    order : accuracy order.  2 = standard central differences,
                             4 = 4th-order scheme (recommended).

    Returns
    -------
    dA : (N, n_modes)  — time derivatives, same shape as A

    Stencils used (4th order)
    -------------------------
    Interior  i ∈ [2, N-3] :  (-A[i+2]+8A[i+1]-8A[i-1]+A[i-2]) / 12h
    i = 1    (one-sided 4th):  (-3A[0]-10A[1]+18A[2]-6A[3]+A[4])  / 12h
    i = 0    (one-sided 4th): (-25A[0]+48A[1]-36A[2]+16A[3]-3A[4]) / 12h
    i = N-2  / i = N-1 : mirror of above (backward)

    Fall-back: if N < 5, falls back to order-2 (central differences).
    """
    A = np.asarray(A, dtype=np.float64)
    N = A.shape[0]
    dA = np.empty_like(A)

    if order == 4 and N >= 5:
        h = dt
        # --- interior: 4th-order central ---
        dA[2:-2] = (-A[4:] + 8.0*A[3:-1] - 8.0*A[1:-3] + A[:-4]) / (12.0 * h)
        # --- i = 0: one-sided forward, 4th order ---
        dA[0]  = (-25.0*A[0] + 48.0*A[1] - 36.0*A[2] + 16.0*A[3] -  3.0*A[4]) / (12.0 * h)
        # --- i = 1: skewed forward, 4th order ---
        dA[1]  = ( -3.0*A[0] - 10.0*A[1] + 18.0*A[2] -  6.0*A[3] +       A[4]) / (12.0 * h)
        # --- i = N-1: one-sided backward, 4th order ---
        dA[-1] = ( 25.0*A[-1] - 48.0*A[-2] + 36.0*A[-3] - 16.0*A[-4] +  3.0*A[-5]) / (12.0 * h)
        # --- i = N-2: skewed backward, 4th order ---
        dA[-2] = (  3.0*A[-1] + 10.0*A[-2] - 18.0*A[-3] +  6.0*A[-4] -       A[-5]) / (12.0 * h)
    else:
        # 2nd-order central differences, 1st-order at boundaries
        dA[1:-1] = (A[2:] - A[:-2]) / (2.0 * dt)
        dA[0]    = (A[1]  - A[0])   / dt
        dA[-1]   = (A[-1] - A[-2])  / dt

    return dA


def build_sindy_library(
    A: np.ndarray,
    poly_order: int = 2,
    include_bias: bool = True,
    include_trig: bool = False,
    include_abs: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build SINDy candidate library from modal coefficient matrix.

    Parameters
    ----------
    A            : (n_samples, n_modes)  — POD modal coefficients
    poly_order   : maximum polynomial degree  (1 = linear, 2 = quadratic, 3 = cubic, …)
    include_bias : prepend constant column ("1")
    include_trig : append sin(aₖ) and cos(aₖ) for every mode k
    include_abs  : append |aₖ| for every mode k

    Returns
    -------
    Theta        : (n_samples, n_features)  — library matrix
    feature_names: list of strings, one per column

    Column ordering
    ---------------
    [1?]  [deg-1 mono]  [deg-2 mono]  …  [deg-p mono]  [sin/cos?]  [|a|?]

    Polynomial monomials for each degree use ``combinations_with_replacement``
    so all mixed terms are included (e.g. a0²·a1 for cubic).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[np.newaxis, :]

    n_samples, n_modes = A.shape
    cols: List[np.ndarray] = []
    names: List[str] = []

    if include_bias:
        cols.append(np.ones((n_samples, 1)))
        names.append("1")

    # Polynomial terms: degrees 1 … poly_order via combinations_with_replacement
    for order in range(1, poly_order + 1):
        for idx in combinations_with_replacement(range(n_modes), order):
            col = np.prod([A[:, i] for i in idx], axis=0).reshape(-1, 1)
            cnt = Counter(idx)
            name = "*".join(
                f"a{i}^{c}" if c > 1 else f"a{i}"
                for i, c in sorted(cnt.items())
            )
            cols.append(col)
            names.append(name)

    # Trigonometric terms: sin(aₖ), cos(aₖ) for each mode
    if include_trig:
        for k in range(n_modes):
            cols.append(np.sin(A[:, k]).reshape(-1, 1))
            names.append(f"sin(a{k})")
            cols.append(np.cos(A[:, k]).reshape(-1, 1))
            names.append(f"cos(a{k})")

    # Absolute-value terms: |aₖ| for each mode
    if include_abs:
        for k in range(n_modes):
            cols.append(np.abs(A[:, k]).reshape(-1, 1))
            names.append(f"|a{k}|")

    Theta = np.concatenate(cols, axis=1)
    return Theta, names


def _stlsq(
    Theta: np.ndarray,
    dA: np.ndarray,
    threshold: float = 0.05,
    regularization: float = 1e-4,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Sequential Thresholded Least Squares (STLSQ).
    Returns Xi: (n_features, n_modes)
    """
    n_features = Theta.shape[1]
    n_modes = dA.shape[1]
    A_reg = Theta.T @ Theta + regularization * np.eye(n_features)
    b = Theta.T @ dA
    Xi = np.linalg.solve(A_reg, b)

    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for m in range(n_modes):
            active = ~small[:, m]
            if active.sum() == 0:
                continue
            Th = Theta[:, active]
            A_r = Th.T @ Th + regularization * np.eye(active.sum())
            Xi[active, m] = np.linalg.solve(A_r, Th.T @ dA[:, m])
    return Xi


def discover_sindy_dynamics(
    A: np.ndarray,
    dA: np.ndarray,
    poly_order: int = 2,
    sparsity_threshold: float = 0.05,
    regularization: float = 1e-4,
    include_bias: bool = True,
    include_trig: bool = False,
    include_abs: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Discover sparse dynamics  dA/dt = f(A)  via STLSQ.

    Accepts NumPy or JAX arrays for A and dA.

    Returns
    -------
    Xi   : (n_features, n_modes)
    info : dict with feature_names, fit_error, sparsity_count,
           sparsity_mask, discovered_equations
    """
    # Coerce to numpy — safe for JAX / torch arrays as well
    A  = np.asarray(A,  dtype=np.float64)
    dA = np.asarray(dA, dtype=np.float64)

    Theta, feature_names = build_sindy_library(
        A, poly_order, include_bias, include_trig, include_abs
    )
    Xi = _stlsq(Theta, dA, sparsity_threshold, regularization)

    dA_pred   = Theta @ Xi
    fit_error = float(np.mean((dA_pred - dA) ** 2))
    mask      = np.abs(Xi) > 0

    equations: Dict[str, str] = {}
    for m in range(A.shape[1]):
        terms = [
            f"{Xi[fi, m]:+.4f}*{feature_names[fi]}"
            for fi in range(len(feature_names)) if mask[fi, m]
        ]
        equations[f"mode_{m}"] = f"da{m}/dt = " + (" ".join(terms) if terms else "0")

    info = {
        "feature_names":        feature_names,
        "fit_error":            fit_error,
        "sparsity_count":       mask.sum(axis=0),
        "sparsity_mask":        mask,
        "discovered_equations": equations,
    }
    return Xi, info


def sindy_integrate(
    Xi: np.ndarray,
    a0: np.ndarray,
    t: np.ndarray,
    poly_order: int = 2,
    include_bias: bool = True,
    include_trig: bool = False,
    include_abs: bool = False,
) -> np.ndarray:
    """
    Integrate the discovered SINDy ODE forward using RK4.

    Returns A_pred : (n_steps, n_modes)
    """
    def rhs(a: np.ndarray) -> np.ndarray:
        th, _ = build_sindy_library(
            a[np.newaxis], poly_order, include_bias, include_trig, include_abs
        )
        return (th @ Xi)[0]

    n_steps = len(t)
    A_pred = np.empty((n_steps, len(a0)))
    A_pred[0] = a0
    for i in range(1, n_steps):
        dt = t[i] - t[i - 1]
        a  = A_pred[i - 1]
        k1 = rhs(a)
        k2 = rhs(a + 0.5 * dt * k1)
        k3 = rhs(a + 0.5 * dt * k2)
        k4 = rhs(a + dt * k3)
        A_pred[i] = a + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return A_pred


# ============================================================================
# SHRED — PyTorch (MPS / CUDA / CPU)
# ============================================================================

class SHRED(nn.Module):
    """
    Shallow Recurrent Decoder.

    Maps a lag-window of sparse sensor readings → POD modal coefficients.

    Architecture: LSTM → last hidden state → MLP decoder → ℝ^n_modes
    """

    def __init__(
        self,
        n_sensors: int,
        n_modes: int,
        hidden_size: int = 64,
        n_lstm_layers: int = 2,
        decoder_sizes: Tuple[int, ...] = (128, 256),
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_sensors, hidden_size, n_lstm_layers,
            batch_first=True,
        )
        layers: List[nn.Module] = []
        in_sz = hidden_size
        for sz in decoder_sizes:
            layers += [nn.Linear(in_sz, sz), activation]
            in_sz = sz
        layers.append(nn.Linear(in_sz, n_modes))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lag, n_sensors)  →  (B, n_modes)"""
        _, (h, _) = self.lstm(x)
        return self.decoder(h[-1])


class SensorMLP(nn.Module):
    """
    Baseline MLP (no recurrence).

    Flattens the lag window of sensor readings and maps directly to POD modes.

    Input  : (B, lag, n_sensors)  →  flattened to (B, lag * n_sensors)
    Output : (B, n_modes)
    """

    def __init__(
        self,
        n_sensors: int,
        n_modes: int,
        lag: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        in_dim = lag * n_sensors
        layers: List[nn.Module] = []
        in_sz = in_dim
        for sz in hidden_sizes:
            layers += [nn.Linear(in_sz, sz), activation]
            in_sz = sz
        layers.append(nn.Linear(in_sz, n_modes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lag, n_sensors)  →  (B, n_modes)"""
        return self.net(x.flatten(1))


def train_shred(
    model: SHRED,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 40,
    sched_patience: int = 20,
) -> Tuple[List[float], List[float]]:
    """
    Train SHRED with Adam + ReduceLROnPlateau + early stopping.

    Loss: MSE with sum over output axis first, then mean over samples
    (i.e. ``mse_fn`` — not ``nn.MSELoss()``).

    Parameters
    ----------
    X_tr/Y_tr      : (N, lag, n_sensors) / (N, n_modes)  — numpy float32
    X_val/Y_val    : validation counterparts
    device         : auto-detected if None
    sched_patience : patience for ReduceLROnPlateau

    Returns
    -------
    train_losses, val_losses
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=sched_patience, factor=0.5
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type in ("cuda", "mps")),
    )
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)

    best_val, best_state, no_improve = np.inf, None, 0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = mse_fn(yb, model(xb))   # sum over modes, mean over batch
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = mse_fn(Yv, model(Xv)).item()
        sched.step(val_loss)
        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep}")
                break

        print(
            f"  Ep {ep:4d}  train={ep_loss:.5f}  val={val_loss:.5f}"
            f"  lr={opt.param_groups[0]['lr']:.1e}"
            f"  no-improve={no_improve}/{patience}",
            end='\r',
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return train_losses, val_losses


# ============================================================================
# SINDy-SHRED joint training (POD-mode regularisation)
# ============================================================================

def _build_sindy_library_torch(
    A: torch.Tensor,
    poly_order: int = 1,
    include_bias: bool = True,
    include_trig: bool = False,
    include_abs: bool = False,
) -> torch.Tensor:
    """
    Differentiable SINDy library in PyTorch — mirrors ``build_sindy_library``
    column-for-column so that a shared Ξ can be used in both.

    Parameters
    ----------
    A : (N, n_modes)
    poly_order   : maximum polynomial degree
    include_bias : prepend constant column
    include_trig : append sin(aₖ), cos(aₖ) for every mode k
    include_abs  : append |aₖ| for every mode k  (not differentiable at 0)

    Returns
    -------
    Theta : (N, n_features)  — same column ordering as ``build_sindy_library``
    """
    N, n_modes = A.shape
    cols: List[torch.Tensor] = []

    if include_bias:
        cols.append(torch.ones(N, 1, dtype=A.dtype, device=A.device))

    # Polynomial terms via combinations_with_replacement
    for order in range(1, poly_order + 1):
        for idx in combinations_with_replacement(range(n_modes), order):
            col = A[:, idx[0]]
            for i in idx[1:]:
                col = col * A[:, i]
            cols.append(col.unsqueeze(-1))

    # Trigonometric terms
    if include_trig:
        for k in range(n_modes):
            cols.append(torch.sin(A[:, k]).unsqueeze(-1))
            cols.append(torch.cos(A[:, k]).unsqueeze(-1))

    # Absolute-value terms  (sub-gradient at 0)
    if include_abs:
        for k in range(n_modes):
            cols.append(A[:, k].abs().unsqueeze(-1))

    return torch.cat(cols, dim=1)


def _refit_xi_on_predictions(
    model: SHRED,
    X_tr_traj: np.ndarray,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    dt: float,
    poly_order: int,
    include_bias: bool,
    sparsity_threshold: float,
    regularization: float,
    device: torch.device,
    inference_batch: int = 2048,
    include_trig: bool = False,
    include_abs: bool = False,
) -> np.ndarray:
    """
    Re-fit SINDy coefficients Ξ on the *current* SHRED modal predictions.

    Uses central finite differences for dA/dt, then STLSQ.

    Parameters
    ----------
    X_tr_traj : (n_train, Nt, lag, n_sensors)

    Returns
    -------
    Xi_new : (n_features, n_modes)
    """
    model.eval()
    n_traj, Nt = X_tr_traj.shape[:2]
    X_flat = X_tr_traj.reshape(n_traj * Nt, *X_tr_traj.shape[2:])

    a_sc_parts = []
    with torch.no_grad():
        for i in range(0, len(X_flat), inference_batch):
            xb = torch.from_numpy(X_flat[i : i + inference_batch]).to(device)
            a_sc_parts.append(model(xb).cpu().numpy())
    a_sc = np.concatenate(a_sc_parts, axis=0)            # (n_traj*Nt, n_modes)
    a    = a_sc * std_np + mean_np                        # unscale

    # 4th-order finite differences per trajectory
    a_traj = a.reshape(n_traj, Nt, -1)
    dA_rows = [
        time_derivative(a_traj[i], dt, order=4)
        for i in range(n_traj)
    ]
    dA = np.vstack(dA_rows)                               # (n_traj*Nt, n_modes)

    Xi_new, _ = discover_sindy_dynamics(
        a, dA,
        poly_order=poly_order,
        sparsity_threshold=sparsity_threshold,
        regularization=regularization,
        include_bias=include_bias,
        include_trig=include_trig,
        include_abs=include_abs,
    )
    return Xi_new


def train_sindy_shred(
    model: SHRED,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_tr_traj: np.ndarray,
    Xi: np.ndarray,
    scaler,
    dt: float,
    poly_order: int = 1,
    include_bias: bool = True,
    include_trig: bool = False,
    include_abs: bool = False,
    sindy_lambda: float = 0.1,
    sindy_xi_lambda: float = 0.05,
    sindy_update_interval: int = 0,
    regularization: float = 1e-4,
    sindy_traj_batch: int = 16,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 40,
    sched_patience: int = 20,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Train SHRED with joint SINDy regularisation on POD modes.

    The total loss per epoch is (matches paper formulation, page 6):

        L = L_recon  +  λ₁ * L_sindy  +  λ₂ * ||Ξ||_0

    where:
        L_recon = MSE(a_pred_scaled, a_true_scaled)       (windowed batch)
        L_sindy = MSE(a_{t+1}, a_t + Θ(a_t)·Ξ·dt)       (trajectory batch, unscaled)
        λ₁      = sindy_lambda — weight on the Euler-consistency term
        λ₂      = sindy_xi_lambda — L0 threshold on Ξ (not differentiable;
                  enforced via STLSQ hard-thresholding every sindy_update_interval epochs)

    ||Ξ||_0 counts non-zero entries of Ξ.  Since it is not differentiable, it is
    minimised by periodically running STLSQ with threshold sindy_xi_lambda, which
    zeros every coefficient |ξ| < sindy_xi_lambda.

    POD modes are **unscaled** before the SINDy consistency check so that
    Ξ carries physical meaning on the original POD basis.

    Ξ is jointly optimised with gradient descent (it is a learnable Parameter).

    Parameters
    ----------
    X_tr, Y_tr       : (N, lag, n_sensors) / (N, n_modes)  shuffled windowed pairs
    X_val, Y_val     : validation counterparts
    X_tr_traj        : (n_train, Nt, lag, n_sensors)  trajectory-ordered windows
    Xi               : (n_features, n_modes)  initial SINDy coefficients (unscaled modes)
    scaler           : fitted sklearn StandardScaler used to normalise Y_tr
    dt               : simulation time step
    sindy_lambda     : λ₁ — weight on the SINDy Euler-consistency loss term
    sindy_xi_lambda  : λ₂ — L0 sparsity threshold for Ξ; coefficients with |ξ| < λ₂
                       are zeroed during each STLSQ re-fit (sindy_update_interval)
    sindy_update_interval : re-fit Ξ via STLSQ every this many epochs (0 = gradient-only)
    sindy_traj_batch : number of training trajectories sampled per epoch for L_sindy
    poly_order, include_bias, include_trig, include_abs :
                     must match the library settings used when Ξ was discovered

    Returns
    -------
    train_losses : List[float]  total loss per epoch
    val_losses   : List[float]  validation reconstruction loss per epoch
    Xi_final     : np.ndarray (n_features, n_modes)  final SINDy coefficients
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    # Scaler parameters as tensors for differentiable unscaling
    mean_t = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)
    std_t  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    mean_np = scaler.mean_.astype(np.float64)
    std_np  = scaler.scale_.astype(np.float64)

    Xi_curr = Xi.astype(np.float64).copy()
    # \u039e is a jointly-optimised learnable parameter (gradients + L1 sparsity reg)
    Xi_t    = nn.Parameter(
        torch.tensor(Xi_curr, dtype=torch.float32, device=device)
    )  # shape (n_features, n_modes)

    n_traj, Nt = X_tr_traj.shape[:2]

    opt   = torch.optim.Adam([*model.parameters(), Xi_t], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=sched_patience, factor=0.5
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type in ("cuda", "mps")),
    )
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)

    best_val, best_state, no_improve = np.inf, None, 0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_recon = 0.0

        # ── Reconstruction batches ────────────────────────────────────────
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = mse_fn(yb, model(xb))
            loss.backward()
            opt.step()
            ep_recon += loss.item() * len(xb)
        ep_recon /= len(X_tr)

        # ── SINDy regularisation on a trajectory sample ───────────────────
        ep_sindy = 0.0
        if sindy_lambda > 0.0:
            traj_idx = np.random.choice(
                n_traj, min(sindy_traj_batch, n_traj), replace=False
            )
            X_traj_b = X_tr_traj[traj_idx]                    # (Bt, Nt, lag, ns)
            Bt       = len(traj_idx)
            X_flat_t = torch.from_numpy(
                X_traj_b.reshape(Bt * Nt, *X_tr_traj.shape[2:])
            ).to(device)

            opt.zero_grad()
            a_sc_flat = model(X_flat_t)                        # (Bt*Nt, n_modes) scaled
            a_flat    = a_sc_flat * std_t + mean_t             # unscale — differentiable
            a_seq     = a_flat.view(Bt, Nt, -1)               # (Bt, Nt, n_modes)

            # Consecutive pairs
            a_t   = a_seq[:, :-1].reshape(-1, a_seq.shape[-1])   # (Bt*(Nt-1), n_modes)
            a_tp1 = a_seq[:, 1: ].reshape(-1, a_seq.shape[-1])   # (Bt*(Nt-1), n_modes)

            # SINDy Euler-consistency — both a_t/a_tp1 (SHRED) and Ξ_t see gradients.
            # L0 sparsity is NOT applied here (not differentiable); it is enforced
            # via STLSQ hard-thresholding every sindy_update_interval epochs.
            Theta_t    = _build_sindy_library_torch(a_t, poly_order, include_bias, include_trig, include_abs)
            a_euler    = a_t + Theta_t @ Xi_t * dt            # Euler step using current Ξ
            sindy_loss = mse_fn(a_tp1, a_euler)
            (sindy_lambda * sindy_loss).backward()
            opt.step()
            ep_sindy = sindy_loss.item()

        ep_total = ep_recon + sindy_lambda * ep_sindy

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = mse_fn(Yv, model(Xv)).item()
        sched.step(val_loss)
        train_losses.append(ep_total)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            best_xi    = Xi_t.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stop at epoch {ep}")
                break

        print(
            f"  Ep {ep:4d}  recon={ep_recon:.5f}  sindy={ep_sindy:.5f}"
            f"  val={val_loss:.5f}  lr={opt.param_groups[0]['lr']:.1e}"
            f"  ({no_improve}/{patience})",
            end='\r',
        )

        # ── Optionally re-fit Ξ via STLSQ ─────────────────────────────────
        if sindy_update_interval > 0 and ep % sindy_update_interval == 0:
            Xi_curr = _refit_xi_on_predictions(
                model, X_tr_traj, mean_np, std_np, dt,
                poly_order, include_bias,
                sindy_xi_lambda, regularization, device,
                include_trig=include_trig, include_abs=include_abs,
            )
            # Update the parameter data in-place so the optimizer keeps tracking it
            Xi_t.data = torch.tensor(Xi_curr, dtype=torch.float32, device=device)
            print(f"\n  [Ξ re-fitted at epoch {ep}  "
                  f"non-zero={(np.abs(Xi_curr) > 0).sum()}]", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        Xi_t.data = best_xi.to(device)
    model.eval()
    print()
    return train_losses, val_losses, Xi_t.detach().cpu().numpy()


def shred_predict(
    model: SHRED,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Batched inference, returns numpy array on CPU.

    X: (N, lag, n_sensors)  →  Y_pred: (N, n_modes)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ============================================================================
# Metric evaluation
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics using the per-sample convention:

        mae     = mean( |y_pred - y_true| )
        mse     = mean( sum_j (y_pred_j - y_true_j)^2 )   ← sum over last axis, mean over samples
        mre     = mean( ||y_pred - y_true||_2 / ||y_true||_2 )   ← per-sample relative L2, then mean
        max_err = max( |y_pred - y_true| )

    Parameters
    ----------
    y_true, y_pred : (N, features) or (N,) arrays

    Returns
    -------
    dict with keys: mae, mse, mre, max_err
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    residual = y_pred - y_true
    mae     = float(np.abs(residual).mean())
    mse     = float((residual ** 2).sum(axis=-1).mean())
    denom   = np.sqrt((y_true ** 2).sum(axis=-1)) + 1e-12
    mre     = float((np.sqrt((residual ** 2).sum(axis=-1)) / denom).mean())
    max_err = float(np.abs(residual).max())
    return dict(mae=mae, mse=mse, mre=mre, max_err=max_err)


def evaluate_reconstruction(
    modes_pred: np.ndarray,
    modes_true: np.ndarray,
    Phi: np.ndarray,
    snaps_true: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics at both the modal-coefficient level and the full-field level.

    Parameters
    ----------
    modes_pred  : (N, n_modes)  predicted POD coefficients  (un-scaled)
    modes_true  : (N, n_modes)  ground-truth POD coefficients
    Phi         : (Nx, n_modes) POD spatial basis
    snaps_true  : (N, Nx) ground-truth full-field snapshots (optional).
                  If provided, field-level metrics are also computed.

    Returns
    -------
    dict with sub-dicts 'modal' and (if snaps_true given) 'field'
    """
    modes_pred = np.asarray(modes_pred, dtype=np.float64)
    modes_true = np.asarray(modes_true, dtype=np.float64)
    results: Dict[str, Dict[str, float]] = {}
    results["modal"] = compute_metrics(modes_true, modes_pred)

    if snaps_true is not None:
        field_pred = modes_pred @ Phi.T   # (N, Nx)
        field_true = np.asarray(snaps_true, dtype=np.float64).reshape(len(modes_pred), -1)
        results["field"] = compute_metrics(field_true, field_pred)
    return results


def print_metrics(
    results: Dict,
    title: str = "",
) -> None:
    """
    Pretty-print the output of evaluate_reconstruction or compute_metrics.
    """
    if title:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
    for level, m in results.items():
        if isinstance(m, dict) and "mse" in m:
            print(f"  [{level}]")
            print(f"    MSE         = {m['mse']:.6e}")
            print(f"    MRE         = {m['mre']*100:.3f} %")
            print(f"    MAE         = {m['mae']:.6e}")
            print(f"    Max error   = {m['max_err']:.6e}")


# ============================================================================
# Combined SINDy-SHRED wrapper
# ============================================================================

class SINDySHRED:
    """
    Combined SINDy-SHRED pipeline:
      sensors → SHRED → POD modes a(t) → SINDy ODE → integrate → reconstruct u(x,t)
    """

    def __init__(
        self,
        shred: SHRED,
        Phi: np.ndarray,
        scaler,
        Xi: Optional[np.ndarray] = None,
        poly_order: int = 2,
        include_bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.shred        = shred
        self.Phi          = Phi
        self.scaler       = scaler
        self.Xi           = Xi
        self.poly_order   = poly_order
        self.include_bias = include_bias
        self.device       = device or get_device()

    def predict_modes(self, X_windows: np.ndarray) -> np.ndarray:
        """SHRED → inverse-scaled POD coefficients. (N, n_modes)"""
        a_sc = shred_predict(self.shred, X_windows, self.device)
        return self.scaler.inverse_transform(a_sc)

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Decode POD coefficients to full field. (N, Nx)"""
        return a @ self.Phi.T

    def forecast(self, X_window0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict full-field trajectory via SINDy integration.

        X_window0 : (lag, n_sensors)  — sensor window at t[0]
        t         : (n_steps,)
        Returns   : (n_steps, Nx)
        """
        if self.Xi is None:
            raise RuntimeError("SINDy coefficients not set. Call discover_sindy_dynamics first.")
        a0_sc  = shred_predict(self.shred, X_window0[np.newaxis], self.device)[0]
        a0     = self.scaler.inverse_transform(a0_sc[np.newaxis])[0]
        A_pred = sindy_integrate(self.Xi, a0, t, self.poly_order, self.include_bias)
        return A_pred @ self.Phi.T
