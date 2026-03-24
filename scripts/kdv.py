"""
KdV equation solver (periodic) using Fourier pseudospectral method + ETDRK4 time-stepping.

Equation:
  u_t + q u u_x + u_xxx = 0
  => u_t = -q u u_x - u_xxx

We split as: u_t = L u + N(u)
  L u = -u_xxx  (in Fourier: L_hat = i*k^3)
  N(u) = 6 u u_x (computed via FFT as 3 i k FFT(u^2))

Time stepping: ETDRK4 with diagonal linear operator (Kassam & Trefethen, 2005).
Dealiasing: 2/3 rule (optional, enabled by default).

Usage:  
  python kdv_solver.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False



@dataclass
class KDVSolution:
    x: np.ndarray            # spatial grid (periodic)
    t: np.ndarray            # times for saved snapshots
    u: np.ndarray            # solution snapshots shape (nt, N)
    Lx: float                # domain length
    N: int                   # number of grid points
    dt: float                # time step
    steps: int               # total time steps
    dealias: bool            # whether dealiasing was used


def _int_wavenumbers(N: int) -> np.ndarray:
    """
    Integer wavenumbers n in standard FFT ordering: [0, 1, ..., N/2, -N/2+1, ..., -1]
    """
    # Using spacing d=1/N yields integer frequencies
    return np.fft.fftfreq(N, d=1.0 / N)


def _etdrk4_coeffs(L: np.ndarray, dt: float, M: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ETDRK4 coefficients for diagonal linear operator L (vector) and timestep dt.
    Returns (E, E2, Q, f1, f2, f3) as vectors of length len(L).
    Based on Kassam & Trefethen (2005).
    """
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # Find zero modes (where L = 0)
    zero_modes = (np.abs(L) < 1e-14)
    
    # Initialize arrays
    Q = np.zeros_like(L, dtype=complex)
    f1 = np.zeros_like(L, dtype=complex)
    f2 = np.zeros_like(L, dtype=complex)
    f3 = np.zeros_like(L, dtype=complex)
    
    # Handle zero modes analytically (L = 0 case)
    Q[zero_modes] = dt
    f1[zero_modes] = dt
    f2[zero_modes] = dt
    f3[zero_modes] = dt
    
    # Handle non-zero modes with contour integration
    nonzero_mask = ~zero_modes
    if np.any(nonzero_mask):
        L_nz = L[nonzero_mask]
        
        # Roots of unity on unit circle for contour integral approximation
        k = np.arange(1, M + 1)
        r = np.exp(1j * np.pi * (k - 0.5) / M)
        
        # Broadcast: (modes, M)
        LR = dt * L_nz[:, None] + r[None, :]
        
        Q[nonzero_mask] = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1)
        
        f1[nonzero_mask] = dt * np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3),
            axis=1,
        )
        f2[nonzero_mask] = dt * np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3),
            axis=1,
        )
        f3[nonzero_mask] = dt * np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3),  
            axis=1,
        )

    return E, E2, Q, f1, f2, f3


def kdv_etdrk4(
    u0: np.ndarray,
    Lx: float,
    dt: float,
    steps: int,
    save_every: int = 10,
    dealias: bool = False,
    M_phi: int = 32,
    progress: bool = True,
    q: float = -6.0,
) -> KDVSolution:
    """
    Solve parametric KdV with periodic BCs using Fourier pseudospectral + ETDRK4.
    
    Equation: u_t + q*u*u_x + u_xxx = 0
    
    Parameters:
      u0         initial condition on grid x (length N), periodic
      Lx         domain length (period is Lx)
      dt         time step
      steps      number of time steps to advance
      save_every save every k steps (including step 0)
      dealias    whether to apply 2/3 de-aliasing on nonlinear term
      M_phi      number of points for phi-function contour integrals
      progress   print progress every ~10%
      q          nonlinear coefficient (default -6.0 for standard KdV)

    Returns:
      KDVSolution with snapshots u at saved times
    """
    u0 = np.asarray(u0, dtype=float)
    N = u0.size
    x = np.linspace(0, Lx, N, endpoint=False)

    # Integer wavenumbers and physical wavenumbers in radians
    n = _int_wavenumbers(N)  # integers
    k = 2.0 * np.pi / Lx * n  # radians per unit length

    # Linear operator in Fourier: L = +i k^3 (since u_t = -u_xxx - 6*u*u_x, linear part is -u_xxx)
    # For u_xxx in Fourier: u_xxx ↔ (ik)³*û = -ik³*û, so -u_xxx ↔ +ik³*û
    L = +1j * (k**3)

    # ETDRK4 coefficients (vectors of length N, complex)
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(L, dt, M=M_phi)

    # Dealias mask: keep |n| <= N/3
    if dealias:
        mask = np.abs(n) <= (N // 3)
    else:
        mask = np.ones_like(n, dtype=bool)

    # Helpers
    fft = np.fft.fft
    ifft = np.fft.ifft

    # Nonlinear term in Fourier space: N_hat(u) = (q/2) * i * k * FFT(u^2) for q*u*u_x
    # Since u*u_x = (1/2)*d(u^2)/dx, we have FFT(q*u*u_x) = q*(1/2)*ik*FFT(u^2) = (q/2)*ik*FFT(u^2)
    def N_hat_from_v(v_hat: np.ndarray) -> np.ndarray:
        u = np.real(ifft(v_hat))
        N_hat = (q / 2.0) * 1j * k * fft(u * u)
        if dealias:
            N_hat = N_hat * mask
        return N_hat

    # Initialize
    v = fft(u0)

    # Saving
    num_saves = steps // save_every + 1
    U = np.empty((num_saves, N), dtype=float)
    T = np.empty(num_saves, dtype=float)

    save_idx = 0
    U[save_idx] = u0
    T[save_idx] = 0.0
    save_idx += 1

    # Time stepping
    next_progress = 0.1
    for m in range(1, steps + 1):
        Nv = N_hat_from_v(v)

        a = E2 * v + Q * Nv
        Na = N_hat_from_v(a)

        b = E2 * v + Q * Na
        Nb = N_hat_from_v(b)

        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = N_hat_from_v(c)

        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc

        # Divergence detection
        u_real = np.real(ifft(v))
        if np.any(np.isnan(u_real)) or np.any(np.isinf(u_real)):
            raise RuntimeError(f"KdV solver diverged at step {m}: NaN or Inf encountered.")
        if np.max(np.abs(u_real)) > 1e12:
            raise RuntimeError(f"KdV solver diverged at step {m}: solution norm exceeded 1e12.")

        if m % save_every == 0:
            U[save_idx] = u_real
            T[save_idx] = m * dt
            save_idx += 1

        if progress and m >= int(next_progress * steps):
            print(f"[KdV] {int(100*next_progress)}% complete")
            next_progress += 0.1

    return KDVSolution(x=x, t=T, u=U, Lx=Lx, N=N, dt=dt, steps=steps, dealias=dealias)


def kdv_soliton(x: np.ndarray, c: float = 1.0, x0: float = 0.0, q: float = -6.0) -> np.ndarray:
    """
    One-soliton solution at t=0 for parametric KdV: u_t + q*u*u_x + u_xxx = 0
    
    For q < 0: u(x,0) = (c/2) * sech^2( sqrt(|c|/2) * (x - x0) )
    For q > 0: u(x,0) = -(c/2) * sech^2( sqrt(|c|/2) * (x - x0) )
    
    Parameters:
      x   spatial grid points
      c   soliton amplitude parameter (c > 0)
      x0  soliton center position
      q   nonlinear coefficient (default -6.0)
    
    For periodic domain, ensure domain is wide enough that tails are small.
    """
    xi = np.sqrt(abs(c)) * 0.5 * (x - x0)
    sech_sq = (1.0 / np.cosh(xi)) ** 2
    
    # For q < 0 (standard): positive amplitude
    # For q > 0: negative amplitude to maintain correct physics
    if q < 0:
        return 0.5 * c * sech_sq
    else:
        return -0.5 * c * sech_sq



def run_example():
    # Domain and discretization
    Lx = 40.0          # domain length
    N = 512            # grid points
    x = np.linspace(0, Lx, N, endpoint=False)

    # Initial condition: single soliton centered at Lx/3 with speed c
    c = 1.0
    u0 = kdv_soliton(x, c=c, x0=Lx / 3.0)

    # Time integration params
    dt = 0.01
    Tfinal = 10.0
    steps = int(np.round(Tfinal / dt))
    save_every = 20

    sol = kdv_etdrk4(
        u0=u0,
        Lx=Lx,
        dt=dt,
        steps=steps,
        save_every=save_every,
        dealias=True,
        M_phi=32,
        progress=True,
    )

    print(f"Done. Saved {len(sol.t)} snapshots on N={sol.N} grid, dt={sol.dt:g}, Lx={sol.Lx:g}")

    if _HAS_MPL:
        # Plot initial and final
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(sol.x, sol.u[0], label=f"t={sol.t[0]:.2f}")
        plt.plot(sol.x, sol.u[-1], label=f"t={sol.t[-1]:.2f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title("KdV (ETDRK4 pseudospectral)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    run_example()