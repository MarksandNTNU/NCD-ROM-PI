"""
Generate nonlinear beam dataset.
Run from project root: .venv/bin/python scripts/generate_nonlinear_beam_dataset.py
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from scipy.interpolate import make_interp_spline
import time


# =============================================================================
# Spatial operators
# =============================================================================

def build_D2_interior(nx, dx):
    n = nx - 2
    main = -2.0 * np.ones(n)
    off1 = 1.0 * np.ones(n - 1)
    return diags([off1, main, off1], [-1, 0, 1], shape=(n, n), format="csc") / dx**2


def build_D4_interior(nx, dx):
    n = nx - 2
    main = 6.0 * np.ones(n)
    off1 = -4.0 * np.ones(n - 1)
    off2 = 1.0 * np.ones(n - 2)
    D4 = diags([off2, off1, main, off1, off2], [-2, -1, 0, 1, 2],
               shape=(n, n), format="lil")
    D4[0, 0] += -1.0
    D4[-1, -1] += -1.0
    return D4.tocsc() / dx**4


# =============================================================================
# Nonlinear beam solver (optimised for batch runs)
# =============================================================================

class NonlinearBeam1D:
    def __init__(self, L, alpha, nx, beta=0.25, gamma=0.5):
        self.L = float(L)
        self.alpha = float(alpha)
        self.nx = int(nx)
        self.beta = beta
        self.gamma = gamma

        self.x = np.linspace(0.0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

        self.D2 = build_D2_interior(self.nx, self.dx)
        self.D4 = build_D4_interior(self.nx, self.dx)

        self.interior_idx = np.arange(1, self.nx - 1)
        self.xi = self.x[self.interior_idx]
        n = len(self.xi)
        self.M = eye(n, format="csc")

        # Pre-compute gradient weights for slope_measure
        self._dx_arr = np.diff(self.x)

    def _restrict(self, full_vec):
        arr = np.asarray(full_vec, dtype=float).copy()
        arr[0] = 0.0
        arr[-1] = 0.0
        return arr[self.interior_idx]

    def _expand(self, u_int):
        full = np.zeros(self.nx, dtype=float)
        full[self.interior_idx] = u_int
        return full

    def _slope_measure(self, u_int):
        u_full = self._expand(u_int)
        wx = np.gradient(u_full, self.x)
        return np.trapezoid(wx**2, self.x)

    def _effective_stiffness(self, S):
        return self.D4 - self.alpha * S * self.D2

    def solve(self, T, dt, u0_fun, v0_fun, q_fun=None,
              store_every=1, tol=1e-8, max_iter=20):
        nt = int(np.ceil(T / dt))
        dt = float(dt)
        beta = self.beta
        gamma = self.gamma
        bdt2 = beta * dt**2

        u = self._restrict(u0_fun(self.x))
        v = self._restrict(v0_fun(self.x))

        S0 = self._slope_measure(u)
        K0 = self._effective_stiffness(S0)
        a = -(K0 @ u)

        n_store = nt // store_every + 1
        U_store = np.empty((n_store, self.nx))
        stretch_store = np.empty(n_store)
        times = np.empty(n_store)

        idx = 0
        U_store[idx] = self._expand(u)
        stretch_store[idx] = S0
        times[idx] = 0.0
        idx += 1

        for n in range(nt):
            u_pred = u + dt * v + dt**2 * (0.5 - beta) * a
            v_pred = v + dt * (1.0 - gamma) * a

            u_guess = u.copy()
            for k in range(max_iter):
                S = self._slope_measure(u_guess)
                K_eff = self._effective_stiffness(S)
                Aeff = self.M + bdt2 * K_eff
                rhs = -(K_eff @ u_pred)
                a_new = splu(Aeff).solve(rhs)
                u_new = u_pred + bdt2 * a_new
                err = np.linalg.norm(u_new - u_guess) / max(np.linalg.norm(u_new), 1e-14)
                u_guess = u_new
                if err < tol:
                    break

            a_new = (u_guess - u_pred) / bdt2
            v_new = v_pred + gamma * dt * a_new
            u, v, a = u_guess, v_new, a_new

            if (n + 1) % store_every == 0 and idx < n_store:
                U_store[idx] = self._expand(u)
                stretch_store[idx] = self._slope_measure(u)
                times[idx] = (n + 1) * dt
                idx += 1

        return times[:idx], U_store[:idx], stretch_store[:idx]


# =============================================================================
# IC generators
# =============================================================================

def u0_gaussian_bumps(x, bumps=3, L=2.0):
    amps = np.random.uniform(0.05, 0.5, size=bumps)
    centres = np.random.uniform(0.15 * L, 0.85 * L, size=bumps)
    sigmas = np.random.uniform(0.06 * L, 0.15 * L, size=bumps)
    return sum(amps[i] * np.exp(-((x - centres[i]) / sigmas[i])**2)
               for i in range(bumps))


def u0_fourier(x, n_modes=4, L=2.0):
    amps = np.random.uniform(0.02, 0.25, size=n_modes)
    modes = np.arange(1, n_modes + 1)
    return sum(amps[i] * np.sin(modes[i] * np.pi * x / L)
               for i in range(n_modes))


def u0_rand_poly(x, n_knots=6, L=2.0):
    xp = np.linspace(0, L, n_knots + 1)
    yp = np.random.uniform(-0.3, 0.3, n_knots + 1)
    yp[0] = 0.0
    yp[-1] = 0.0
    cs = make_interp_spline(xp, yp, bc_type="natural", k=3)
    return cs(x).astype(np.float64)


def v0_zero(x):
    return np.zeros_like(x)


def v0_modal(x, L=2.0):
    mode = np.random.randint(1, 4)
    amp = np.random.uniform(0.01, 0.1)
    return amp * np.sin(mode * np.pi * x / L)


# =============================================================================
# Main dataset generation
# =============================================================================

if __name__ == "__main__":
    N_TRAJ = 200
    T_DS = 0.5
    DT_DS = 5e-4
    STORE_EVERY = 5       # → 200 snapshots per trajectory
    NX_DS = 101
    L_DS = 2.0

    np.random.seed(42)

    dataset_U = []
    dataset_stretch = []
    dataset_params = []
    ic_types = ["gaussian", "fourier", "polynomial"]

    t_start = time.time()

    for traj in range(N_TRAJ):
        alpha_traj = np.exp(np.random.uniform(np.log(50), np.log(500)))

        ic_choice = np.random.choice(ic_types)
        n_bumps = np.random.randint(1, 6)

        if ic_choice == "gaussian":
            u0 = lambda x, nb=n_bumps: u0_gaussian_bumps(x, bumps=nb, L=L_DS)
        elif ic_choice == "fourier":
            u0 = lambda x, nm=n_bumps: u0_fourier(x, n_modes=max(nm, 2), L=L_DS)
        else:
            u0 = lambda x, nk=n_bumps: u0_rand_poly(x, n_knots=max(nk, 4), L=L_DS)

        if np.random.random() < 0.3:
            v0 = lambda x: v0_modal(x, L=L_DS)
        else:
            v0 = v0_zero

        solver = NonlinearBeam1D(L=L_DS, alpha=alpha_traj, nx=NX_DS)

        try:
            times, U_ds, stretch_ds = solver.solve(
                T=T_DS, dt=DT_DS,
                u0_fun=u0, v0_fun=v0,
                store_every=STORE_EVERY,
                tol=1e-8, max_iter=20,
            )
            dataset_U.append(U_ds)
            dataset_stretch.append(stretch_ds)
            dataset_params.append((alpha_traj, ic_choice, n_bumps))
        except Exception as e:
            print(f"  Trajectory {traj+1} FAILED (alpha={alpha_traj:.1f}, IC={ic_choice}): {e}")
            continue

        if (traj + 1) % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (traj + 1) * (N_TRAJ - traj - 1)
            print(f"  [{traj+1:3d}/{N_TRAJ}]  alpha={alpha_traj:6.1f}  IC={ic_choice:10s}  "
                  f"max|w|={np.max(np.abs(U_ds)):.4f}  "
                  f"max_S={np.max(stretch_ds):.4f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    dataset_U = np.array(dataset_U)
    dataset_stretch = np.array(dataset_stretch)
    alphas = np.array([p[0] for p in dataset_params])
    ic_labels = np.array([p[1] for p in dataset_params])
    n_bumps_arr = np.array([p[2] for p in dataset_params])

    save_path = "notebooks/nonlinear_beam_dataset.npz"
    np.savez_compressed(
        save_path,
        U=dataset_U,
        times=times,
        x=np.linspace(0, L_DS, NX_DS),
        stretch=dataset_stretch,
        alpha=alphas,
        ic_type=ic_labels,
        n_bumps=n_bumps_arr,
        L=L_DS,
        nx=NX_DS,
        dt=DT_DS,
        T=T_DS,
        store_every=STORE_EVERY,
    )

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s")
    print(f"Dataset saved to {save_path}")
    print(f"  U shape:       {dataset_U.shape}")
    print(f"  times shape:   {times.shape}")
    print(f"  alpha range:   [{alphas.min():.1f}, {alphas.max():.1f}]")
    print(f"  stretch range: [{dataset_stretch.min():.6f}, {dataset_stretch.max():.4f}]")
    print(f"  IC types:      {dict(zip(*np.unique(ic_labels, return_counts=True)))}")
