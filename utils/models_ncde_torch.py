"""
Neural Controlled Differential Equations — PyTorch implementation.

Uses the `torchcde` library (https://github.com/patrick-kidger/torchcde)
for Hermite cubic-spline interpolation and the CDE solver backend.

Main public API
---------------
NeuralCDE          – seq2seq or final-state model
prepare_data_CDE   – build torchcde-ready coefficients from sensor data
fit_CDE            – epoch-based training loop with early stopping
predict_CDE        – inference (returns numpy array)
compute_metrics    – MSE / RMSE / MAE dict (same interface as models_s4)
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchcde
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CDEFunc(nn.Module):
    """Vector field  f(t, z)  with output shape (hidden, data_size)."""

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.data_size = data_size  # full channels including time

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width_size), activation]
            in_dim = width_size
        layers.append(nn.Linear(in_dim, hidden_size * data_size))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, hidden_size)  →  output: (batch, hidden_size, data_size)
        out = self.net(z)
        return out.view(z.shape[0], self.hidden_size, self.data_size)


class _MLP(nn.Module):
    """Fully-connected MLP with configurable hidden layers."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: nn.Module,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = in_size
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width_size), activation]
            in_dim = width_size
        layers.append(nn.Linear(in_dim, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NeuralCDE model
# ---------------------------------------------------------------------------

class NeuralCDE(nn.Module):
    """
    Neural Controlled Differential Equation model.

    Works as a drop-in replacement for the JAX/diffrax ``NeuralCDE`` in
    ``utils/models_diffrax.py``, but runs entirely in PyTorch.

    Parameters
    ----------
    data_size       : number of input channels (sensors, NOT including time)
    hidden_size     : CDE hidden state dimension
    width_size      : MLP hidden width
    depth           : MLP depth
    output_size     : number of output channels (e.g. POD modes)
    decoder_sizes   : list of hidden widths for the decoder MLP
    activation_cde  : activation for the CDE vector field & initial-map MLPs
    activation_dec  : activation for the decoder MLP
    seq2seq         : if True, decode at every time step; else decode only at t1
    adjoint         : if True, use adjoint-based memory-efficient backprop
    rtol, atol      : ODE solver tolerances
    """

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        output_size: int,
        decoder_sizes: Sequence[int] = (64,),
        activation_cde: nn.Module = nn.Tanh(),
        activation_dec: nn.Module = nn.Tanh(),
        seq2seq: bool = True,
        adjoint: bool = False,
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ):
        super().__init__()
        self.seq2seq = seq2seq
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

        # torchcde.CubicSpline.evaluate() returns ALL channels including the
        # appended time column, so the CDE vector field dimension is data_size+1.
        # `data_size` here follows the JAX convention: number of sensors WITHOUT time.
        cde_channels = data_size + 1  # sensors + time

        # Initial hidden state: maps x(t_0) [all channels incl. time] → z_0
        self.initial = _MLP(
            cde_channels, hidden_size, width_size, depth,
            activation=activation_cde,
            final_activation=activation_cde,
        )

        # CDE vector field outputs (hidden, cde_channels)
        self.func = _CDEFunc(cde_channels, hidden_size, width_size, depth, activation_cde)

        # Decoder MLP: hidden_size → output_size
        dec_layers: list[nn.Module] = []
        in_dim = hidden_size
        for w in decoder_sizes:
            dec_layers += [nn.Linear(in_dim, w), activation_dec]
            in_dim = w
        dec_layers.append(nn.Linear(in_dim, output_size))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coeffs : torch.Tensor of shape (batch, length, channels*4)
            Pre-computed Hermite cubic-spline coefficients from
            ``torchcde.hermite_cubic_coefficients_with_backward_differences``.
            The last channel must be time.

        Returns
        -------
        torch.Tensor
            If seq2seq=True : (batch, length, output_size)
            Else            : (batch, output_size)
        """
        X = torchcde.CubicSpline(coeffs)
        z0 = self.initial(X.evaluate(X.interval[0]))

        solve_fn = torchcde.cdeint_adjoint if self.adjoint else torchcde.cdeint

        if self.seq2seq:
            t = X.grid_points
            z = solve_fn(
                X=X,
                func=self.func,
                z0=z0,
                t=t,
                rtol=self.rtol,
                atol=self.atol,
                method="rk4",
            )
            # z: (batch, length, hidden_size)
            return self.decoder(z)

        z1 = solve_fn(
            X=X,
            func=self.func,
            z0=z0,
            t=X.interval,
            rtol=self.rtol,
            atol=self.atol,
            method="rk4",
        )
        # z1: (batch, 2, hidden_size) — take the final state at t1
        return self.decoder(z1[:, -1])


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Return a torch.device, auto-selecting MPS > CUDA > CPU if None."""
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_data_CDE(
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device | str | None = None,
) -> tuple[dict, int]:
    """
    Build torchcde-ready dataset from raw sensor array.

    Parameters
    ----------
    X : np.ndarray, shape (n_traj, n_timesteps, n_sensors + 1)
        Last channel MUST be time.
    Y : np.ndarray, shape (n_traj, n_timesteps, n_outputs) or (n_traj, n_outputs)
    device : torch device (None = auto-select MPS > CUDA > CPU)

    Returns
    -------
    data_dict : dict with keys 'coeffs', 'Y'
    data_size : int  (n_sensors, NOT including time)
    """
    dev = _resolve_device(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)

    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_t)
    data_size = X.shape[-1] - 1  # exclude the time channel
    return {"coeffs": coeffs, "Y": Y_t}, data_size


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fit_CDE(
    model: NeuralCDE,
    train_data: dict,
    val_data: dict,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 20,
    lr_patience: int = 10,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    device: torch.device | str | None = None,
    verbose: bool = True,
    seed: int | None = None,
) -> tuple[NeuralCDE, list[float], list[float]]:
    """
    Epoch-based training with early stopping and LR scheduling.

    Parameters
    ----------
    model      : NeuralCDE instance
    train_data : dict from prepare_data_CDE  (keys: 'coeffs', 'Y')
    val_data   : dict from prepare_data_CDE
    epochs     : max number of epochs
    batch_size : mini-batch size
    lr         : initial learning rate
    patience   : early-stopping patience (epochs without val improvement)
    lr_patience: ReduceLROnPlateau patience
    weight_decay: AdamW weight decay
    grad_clip  : gradient clipping norm
    device     : override device (defaults to model's current device)
    verbose    : print progress

    Returns
    -------
    model       : trained model (best validation checkpoint)
    tr_losses   : list of per-epoch training losses
    vl_losses   : list of per-epoch validation losses
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model = model.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    # Move data to device
    coeffs_tr = train_data["coeffs"].to(device)
    Y_tr      = train_data["Y"].to(device)
    coeffs_vl = val_data["coeffs"].to(device)
    Y_vl      = val_data["Y"].to(device)

    n_train = coeffs_tr.shape[0]
    train_ds = TensorDataset(coeffs_tr, Y_tr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(coeffs_vl, Y_vl)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=lr_patience, factor=0.5)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
    tr_losses: list[float] = []
    vl_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        tr_sum = 0.0
        for c_b, y_b in train_dl:
            pred = model(c_b)
            loss = _mse(pred, y_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_sum += loss.item()
        tr_loss = tr_sum / max(len(train_dl), 1)

        # --- validate ---
        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for c_b, y_b in val_dl:
                vl_sum += _mse(model(c_b), y_b).item()
        vl_loss = vl_sum / max(len(val_dl), 1)

        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        sched.step(vl_loss)

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train {tr_loss:.4e} | Val {vl_loss:.4e} | "
                f"{elapsed:.1f}s | patience {wait}/{patience}",
                end="\r",
            )

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}  (best val {best_val:.4e})")
            break

    if verbose:
        print()

    model.load_state_dict(best_state)
    return model, tr_losses, vl_losses


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_CDE(
    model: NeuralCDE,
    data: dict,
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Run inference on a dataset dict (from prepare_data_CDE).

    Returns
    -------
    np.ndarray of shape (n_traj, length, output_size)  [seq2seq]
           or shape (n_traj, output_size)               [final-state]
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model.eval()

    coeffs = data["coeffs"].to(device)
    n = coeffs.shape[0]
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            c_b = coeffs[start : start + batch_size]
            # .detach().cpu() before .numpy() — required for MPS and CUDA tensors
            preds.append(model(c_b).detach().cpu())
    return torch.cat(preds, dim=0).numpy()


# ---------------------------------------------------------------------------
# Metrics (same interface as models_s4.compute_metrics)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return MSE, RMSE, MAE averaged over all elements."""
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
    mse  = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}
