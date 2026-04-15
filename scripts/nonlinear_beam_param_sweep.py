"""
Parameter sweep to find optimal nonlinearity settings for the beam solver.
Tests different alpha values and initial condition amplitudes.
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


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
    D4 = diags([off2, off1, main, off1, off2], [-2, -1, 0, 1, 2], shape=(n, n), format="lil")
    D4[0, 0] += -1.0
    D4[-1, -1] += -1.0
    return D4.tocsc() / dx**4


L = 2.0
nx = 201
x = np.linspace(0, L, nx)
dx = x[1] - x[0]
D2 = build_D2_interior(nx, dx)
D4 = build_D4_interior(nx, dx)
M = eye(nx - 2, format="csc")
interior_idx = np.arange(1, nx - 1)
xi = x[interior_idx]


def expand(u_int):
    full = np.zeros(nx)
    full[interior_idx] = u_int
    return full


def restrict(full):
    full = np.array(full, dtype=float).copy()
    full[0] = 0.0
    full[-1] = 0.0
    return full[interior_idx]


def slope_measure(u_int):
    u_full = expand(u_int)
    wx = np.gradient(u_full, x)
    return np.trapz(wx**2, x)


def solve_beam(alpha, u0_fun, T=0.3, dt=1e-4, store_every=100):
    beta_nm, gamma_nm = 0.25, 0.5
    nt = int(np.ceil(T / dt))
    u = restrict(u0_fun(x))
    v = np.zeros_like(u)

    S0 = slope_measure(u)
    a = -(D4 @ u - alpha * S0 * (D2 @ u))

    times, U_store, stretch_store = [0.0], [expand(u)], [S0]

    for n in range(nt):
        t_np1 = (n + 1) * dt
        f_np1 = np.zeros_like(u)
        u_pred = u + dt * v + dt**2 * (0.5 - beta_nm) * a
        v_pred = v + dt * (1.0 - gamma_nm) * a

        u_guess = u.copy()
        converged = True
        for k in range(30):
            S = slope_measure(u_guess)
            K_eff_nl = D4 - alpha * S * D2
            Aeff = M + beta_nm * dt**2 * K_eff_nl
            rhs = f_np1 - K_eff_nl @ u_pred
            a_new = splu(Aeff.tocsc()).solve(rhs)
            u_new = u_pred + beta_nm * dt**2 * a_new
            err = np.linalg.norm(u_new - u_guess) / max(np.linalg.norm(u_new), 1e-14)
            u_guess = u_new
            if err < 1e-8:
                break
        else:
            converged = False

        a_new = (u_guess - u_pred) / (beta_nm * dt**2)
        v_new = v_pred + gamma_nm * dt * a_new
        u, v, a = u_guess, v_new, a_new

        if (n + 1) % store_every == 0:
            times.append(t_np1)
            U_store.append(expand(u))
            stretch_store.append(slope_measure(u))

    return np.array(times), np.array(U_store), np.array(stretch_store)


print("=" * 80)
print("Parameter sweep: nonlinear beam")
print("=" * 80)
print(f"{'alpha':>6} | {'amp':>5} | {'max_S':>10} | {'max_disp':>10} | {'rel_diff':>10} | {'status'}")
print("-" * 75)

for alpha in [10, 50, 100, 200, 500, 1000]:
    for amp_scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        u0 = lambda x, a=amp_scale: a * np.sin(np.pi * x / L)
        try:
            t_nl, U_nl, S_nl = solve_beam(alpha, u0)
            t_lin, U_lin, _ = solve_beam(0.0, u0)
            mid = nx // 2
            diff = np.max(np.abs(U_nl[:, mid] - U_lin[:, mid])) / max(np.max(np.abs(U_lin[:, mid])), 1e-14)

            # Energy check: if max displacement grows > 2x initial, might be unstable
            max_disp = np.max(np.abs(U_nl))
            init_max = amp_scale
            status = "OK" if max_disp < 2 * init_max else "LARGE"

            print(f"{alpha:6.0f} | {amp_scale:5.2f} | {np.max(S_nl):10.4f} | {max_disp:10.4f} | {diff:10.4f} | {status}")
        except Exception as e:
            print(f"{alpha:6.0f} | {amp_scale:5.2f} | {'FAILED':>10} | {str(e)[:40]}")

print()
print("rel_diff = max|nonlinear - linear| / max|linear| at midspan")
print("Higher rel_diff -> stronger nonlinear effect")
print("Look for rel_diff > 0.1 (10% deviation from linear) for good nonlinearity")
