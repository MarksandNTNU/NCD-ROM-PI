"""
transport_data.py
-----------------
Data-preparation utilities for the periodic-transport SINDy-SHRED demo.

Public API
----------
solve_transport_analytical(u0, c, x, t) -> U  (Nt, Nx)
ic_gaussian_bumps(x, n_bumps, rng)     -> u0 (Nx,)
ic_fourier_modes(x, Kmax, rng)         -> u0 (Nx,)
ic_mixed(x, rng)                       -> u0 (Nx,)
generate_trajectories(...)             -> all_snaps (n_total, Nt, Nx), all_ic, speeds
compute_pod(snaps_train, ...)          -> Phi, S, S_full
central_diff(A, dt)                    -> dA  (finite-difference time derivative)
make_shred_pairs(snaps, modes_sc, sensor_pos, lag)
                                       -> X  (N, lag, n_sensors), Y (N, n_modes)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from sklearn.utils.extmath import randomized_svd


# ============================================================================
# Analytical solver
# ============================================================================

def solve_transport_analytical(
    u0: np.ndarray,
    c: float,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Exact solution of u_t + c*u_x = 0 on the periodic domain [0, L].

    u(x, t) = u0((x - c*t) mod L)

    Parameters
    ----------
    u0  : (Nx,)  initial condition evaluated on x
    c   : advection speed
    x   : (Nx,)  spatial grid (assumed uniform, periodic)
    t   : (Nt,)  time grid

    Returns
    -------
    U   : (Nt, Nx)  — rows are time steps, columns are spatial grid
    """
    L  = x[-1] - x[0] + (x[1] - x[0])   # domain length (one period)
    U  = np.empty((len(t), len(x)), dtype=np.float64)
    for k, tk in enumerate(t):
        x_shifted = (x - c * tk) % L
        U[k, :]   = np.interp(x_shifted, x % L, u0, period=L)
    return U


# ============================================================================
# Initial-condition generators
# ============================================================================

def ic_gaussian_bumps(
    x: np.ndarray,
    n_bumps: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Periodic Gaussian-bump initial condition.

    Enforces periodicity by summing over 3 images (shift in {-L, 0, +L}).
    """
    if rng is None:
        rng = np.random.default_rng()
    L   = x[-1] - x[0] + (x[1] - x[0])
    u0  = np.zeros_like(x, dtype=np.float64)
    for _ in range(n_bumps):
        mu  = rng.uniform(0, L)
        sig = rng.uniform(0.25, 1)
        amp = rng.uniform(0.5, 1.5)
        for shift in [-1, 0, 1]:
            u0 += amp * np.exp(-0.5 * ((x - mu - shift * L) / sig) ** 2)
    # normalise to [0, 1]
    u0 -= u0.min()
    mx = u0.max()
    if mx > 0:
        u0 /= mx
    return u0


def ic_fourier_modes(
    x: np.ndarray,
    Kmax: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random sum of Fourier modes — exactly periodic by construction.
    """
    if rng is None:
        rng = np.random.default_rng()
    L  = x[-1] - x[0] + (x[1] - x[0])
    u0 = np.zeros_like(x, dtype=np.float64)
    for k in range(1, Kmax + 1):
        amp   = rng.uniform(0.1, 1.0) / k
        phase = rng.uniform(0, 2 * np.pi)
        u0   += amp * np.sin(2 * np.pi * k * x / L + phase)
    u0 -= u0.min()
    mx = u0.max()
    if mx > 0:
        u0 /= mx
    return u0


def ic_mixed(
    x: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Blend of Gaussian bumps and Fourier modes.
    """
    if rng is None:
        rng = np.random.default_rng()
    w  = rng.uniform(0.1, 0.9)
    u0 = (
        w * ic_gaussian_bumps(x, n_bumps=rng.integers(1, 4), rng=rng)
        + (1 - w) * ic_fourier_modes(x, Kmax=rng.integers(2, 7), rng=rng)
    )
    u0 -= u0.min()
    mx = u0.max()
    if mx > 0:
        u0 /= mx
    return u0


# ============================================================================
# Trajectory generation
# ============================================================================

def generate_trajectories(
    n_total: int = 100,
    Nx: int = 128,
    Nt: int = 200,
    T: float = 6.0,
    c_range: Tuple[float, float] = (0.3, 2.5),
    ic_types: Tuple[str, ...] = ("gaussian", "fourier", "mixed"),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an ensemble of transport-equation trajectories.

    Returns
    -------
    all_snaps : (n_total, Nt, Nx)  full-field snapshots
    all_ic    : (n_total, Nx)      initial conditions
    speeds    : (n_total,)         advection speeds
    """
    rng    = np.random.default_rng(seed)
    L      = 2 * np.pi
    x      = np.linspace(0, L, Nx, endpoint=False)
    t      = np.linspace(0, T, Nt)
    speeds = rng.uniform(*c_range, size=n_total)

    all_snaps = np.empty((n_total, Nt, Nx), dtype=np.float32)
    all_ic    = np.empty((n_total, Nx), dtype=np.float32)

    ic_map = {
        "gaussian": ic_gaussian_bumps,
        "fourier":  ic_fourier_modes,
        "mixed":    ic_mixed,
    }

    for i in range(n_total):
        ic_type = ic_types[i % len(ic_types)]
        u0      = ic_map[ic_type](x, rng=rng)
        U       = solve_transport_analytical(u0, float(speeds[i]), x, t)  # (Nt, Nx)
        all_snaps[i] = U.astype(np.float32)   # (Nt, Nx)
        all_ic[i]    = u0.astype(np.float32)

    return all_snaps, all_ic, speeds


# ============================================================================
# POD (Proper Orthogonal Decomposition)
# ============================================================================

def compute_pod(
    snaps_train: np.ndarray,
    n_svd: int = 50,
    energy_threshold: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute POD basis via randomized SVD on training snapshots.

    Parameters
    ----------
    snaps_train     : (n_train, Nt, Nx)  — rows-of-snapshots convention
    n_svd           : number of SVD components to compute
    energy_threshold: keep modes explaining this fraction of energy

    Returns
    -------
    Phi    : (Nx, n_modes)  POD spatial basis  (right singular vectors)
    S      : (n_modes,)     retained singular values
    S_full : (n_svd,)       all computed singular values  (for scree plot)
    """
    n_train, Nt, Nx = snaps_train.shape
    # flatten to (n_train*Nt, Nx)  — each row is one space-time snapshot
    flat = snaps_train.reshape(-1, Nx).astype(np.float64)

    _, S_full, Vt = randomized_svd(flat, n_components=n_svd, random_state=0)
    # Vt: (n_svd, Nx) — right singular vectors are the spatial modes

    # determine number of modes for desired energy
    cumulative = np.cumsum(S_full ** 2) / np.sum(S_full ** 2)
    n_modes    = int(np.searchsorted(cumulative, energy_threshold) + 1)

    Phi = Vt[:n_modes].T   # (Nx, n_modes)
    S   = S_full[:n_modes]
    return Phi, S, S_full


# ============================================================================
# Time derivative (central differences)
# ============================================================================

def central_diff(A: np.ndarray, dt: float) -> np.ndarray:
    """
    Second-order finite-difference time derivative.

    Parameters
    ----------
    A  : (n_steps, n_modes)  or (n_traj, n_steps, n_modes)
    dt : time step size

    Returns
    -------
    dA : same shape as A
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 2:
        dA = np.empty_like(A)
        dA[1:-1]  = (A[2:] - A[:-2]) / (2 * dt)
        dA[0]     = (-3 * A[0] + 4 * A[1] - A[2]) / (2 * dt)
        dA[-1]    = (3 * A[-1] - 4 * A[-2] + A[-3]) / (2 * dt)
    elif A.ndim == 3:
        dA = np.empty_like(A)
        dA[:, 1:-1]  = (A[:, 2:] - A[:, :-2]) / (2 * dt)
        dA[:, 0]     = (-3 * A[:, 0] + 4 * A[:, 1] - A[:, 2]) / (2 * dt)
        dA[:, -1]    = (3 * A[:, -1] - 4 * A[:, -2] + A[:, -3]) / (2 * dt)
    else:
        raise ValueError("A must be 2-D or 3-D")
    return dA


# ============================================================================
# SHRED sliding-window sensor data
# ============================================================================

def make_shred_pairs(
    snaps: np.ndarray,
    modes_sc: np.ndarray,
    sensor_pos: np.ndarray,
    lag: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) pairs for SHRED training via a sliding window.

    Parameters
    ----------
    snaps      : (n_traj, Nt, Nx) float32  — rows-of-snapshots convention
    modes_sc   : (n_traj, Nt, n_modes) scaled modal coefficients float32
    sensor_pos : (n_sensors,) int  spatial sensor indices
    lag        : history window length

    Returns
    -------
    X          : (N, lag, n_sensors)  float32
    Y          : (N, n_modes)         float32
    """
    n_traj, Nt, Nx = snaps.shape
    n_modes        = modes_sc.shape[-1]
    n_sensors      = len(sensor_pos)

    samples_per_traj = Nt - lag
    N   = n_traj * samples_per_traj
    X   = np.empty((N, lag, n_sensors), dtype=np.float32)
    Y   = np.empty((N, n_modes),        dtype=np.float32)

    idx = 0
    for i in range(n_traj):
        sensors = snaps[i, :, sensor_pos]   # (Nt, n_sensors)
        for t in range(samples_per_traj):
            X[idx] = sensors[t : t + lag]
            Y[idx] = modes_sc[i, t + lag]
            idx   += 1
    return X, Y
