"""
Add dataset generation cells to the nonlinear beam notebook.
"""
import json
import uuid

with open('notebooks/nonlinear_beam_solver_1d_notebook.ipynb') as f:
    nb = json.load(f)

def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_lines,
    }

def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source_lines,
    }


# ---- Cell 1: Markdown header ----
cell_md_header = make_md_cell([
    "## 13. Nonlinear beam dataset generation\n",
    "\n",
    "### Parameter choices for enhanced nonlinearity\n",
    "\n",
    "A parameter sweep confirmed that the **nonlinear term** $\\alpha \\int_0^L w_x^2\\,dx \\cdot w_{xx}$ becomes significant when:\n",
    "\n",
    "| Parameter | Range | Effect |\n",
    "|-----------|-------|--------|\n",
    "| $\\alpha$ | 50 – 500 | Controls coupling strength; larger $\\alpha$ amplifies the membrane stiffening |\n",
    "| Amplitude | 0.1 – 0.5 | Larger transverse displacement → steeper slopes → larger $S(u)$ |\n",
    "| IC shape | Multi-bump Gaussians, Fourier sums | Steeper spatial gradients enhance $w_x^2$ |\n",
    "\n",
    "At $\\alpha = 100$, $A_0 = 0.2$: the midspan response deviates by **~64%** from the linear solution.  \n",
    "At $\\alpha = 500$, $A_0 = 0.1$: deviation is **~76%** — strong nonlinear stiffening even at moderate amplitude.\n",
    "\n",
    "The dataset below samples random ICs and $\\alpha$ values in these ranges to produce trajectories with **clearly nonlinear dynamics**.\n",
])

# ---- Cell 2: IC generators ----
cell_ic_generators = make_code_cell([
    "from scipy.interpolate import make_interp_spline\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# Initial condition generators (higher amplitudes for nonlinear regime)\n",
    "# ---------------------------------------------------------------------------\n",
    "\n",
    "def u0_fun_gaussian_bumps(x, bumps=3, L=2.0):\n",
    "    \"\"\"Sum of random Gaussian bumps with amplitudes in [0.05, 0.5].\"\"\"\n",
    "    amps = np.random.uniform(0.05, 0.5, size=bumps)\n",
    "    centres = np.random.uniform(0.15 * L, 0.85 * L, size=bumps)\n",
    "    sigmas = np.random.uniform(0.06 * L, 0.15 * L, size=bumps)\n",
    "    return sum(\n",
    "        amps[i] * np.exp(-((x - centres[i]) / sigmas[i]) ** 2)\n",
    "        for i in range(bumps)\n",
    "    )\n",
    "\n",
    "\n",
    "def u0_fun_fourier_series(x, n_modes=4, L=2.0):\n",
    "    \"\"\"Random Fourier sine series satisfying simply-supported BCs.\"\"\"\n",
    "    amps = np.random.uniform(0.02, 0.25, size=n_modes)\n",
    "    modes = np.arange(1, n_modes + 1)\n",
    "    return sum(\n",
    "        amps[i] * np.sin(modes[i] * np.pi * x / L)\n",
    "        for i in range(n_modes)\n",
    "    )\n",
    "\n",
    "\n",
    "def u0_fun_rand_poly(x, n_knots=6, L=2.0):\n",
    "    \"\"\"Random cubic spline through knots with zero end values.\"\"\"\n",
    "    xp = np.linspace(0, L, n_knots + 1)\n",
    "    yp = np.random.uniform(-0.3, 0.3, n_knots + 1)\n",
    "    yp[0] = 0.0\n",
    "    yp[-1] = 0.0\n",
    "    cs = make_interp_spline(xp, yp, bc_type='natural', k=3)\n",
    "    return cs(x).astype(np.float64)\n",
    "\n",
    "\n",
    "def v0_fun_zero(x):\n",
    "    return np.zeros_like(x)\n",
    "\n",
    "\n",
    "def v0_fun_modal(x, L=2.0):\n",
    "    \"\"\"Small initial velocity with a random low-mode shape.\"\"\"\n",
    "    mode = np.random.randint(1, 4)\n",
    "    amp = np.random.uniform(0.01, 0.1)\n",
    "    return amp * np.sin(mode * np.pi * x / L)\n",
    "\n",
    "\n",
    "print('IC generators ready.')\n",
])

# ---- Cell 3: Markdown for dataset loop ----
cell_md_loop = make_md_cell([
    "### Generate multi-trajectory nonlinear beam dataset\n",
    "\n",
    "Each trajectory uses:\n",
    "- a random $\\alpha \\in [50, 500]$ (log-uniform)\n",
    "- a random IC type (Gaussian bumps, Fourier, or spline)\n",
    "- 1–5 bumps / modes\n",
    "- optional small initial velocity\n",
    "- `T = 1.0`, `dt = 1e-4`, `store_every = 10`\n",
])

# ---- Cell 4: Dataset generation loop ----
cell_dataset_loop = make_code_cell([
    "# ---------------------------------------------------------------------------\n",
    "# Dataset generation\n",
    "# ---------------------------------------------------------------------------\n",
    "\n",
    "N_TRAJ = 200         # number of trajectories\n",
    "T_DS = 1.0           # time horizon per trajectory\n",
    "DT_DS = 1e-4         # time step\n",
    "STORE_EVERY = 10     # store every Nth step → 1000 snapshots per traj\n",
    "NX_DS = 201          # spatial grid points\n",
    "L_DS = 2.0           # beam length\n",
    "\n",
    "np.random.seed(42)\n",
    "\n",
    "dataset_U = []\n",
    "dataset_stretch = []\n",
    "dataset_params = []  # (alpha, ic_type, n_bumps)\n",
    "\n",
    "ic_types = ['gaussian', 'fourier', 'polynomial']\n",
    "\n",
    "for traj in range(N_TRAJ):\n",
    "    # Random alpha in [50, 500] (log-uniform for good coverage)\n",
    "    alpha_traj = np.exp(np.random.uniform(np.log(50), np.log(500)))\n",
    "\n",
    "    # Random IC\n",
    "    ic_choice = np.random.choice(ic_types)\n",
    "    n_bumps = np.random.randint(1, 6)\n",
    "\n",
    "    if ic_choice == 'gaussian':\n",
    "        u0 = lambda x, nb=n_bumps: u0_fun_gaussian_bumps(x, bumps=nb, L=L_DS)\n",
    "    elif ic_choice == 'fourier':\n",
    "        u0 = lambda x, nm=n_bumps: u0_fun_fourier_series(x, n_modes=max(nm, 2), L=L_DS)\n",
    "    else:\n",
    "        u0 = lambda x, nk=n_bumps: u0_fun_rand_poly(x, n_knots=max(nk, 4), L=L_DS)\n",
    "\n",
    "    # 30% chance of non-zero initial velocity\n",
    "    if np.random.random() < 0.3:\n",
    "        v0 = lambda x: v0_fun_modal(x, L=L_DS)\n",
    "    else:\n",
    "        v0 = v0_fun_zero\n",
    "\n",
    "    solver_ds = NonlinearBeam1D(\n",
    "        L=L_DS, alpha=alpha_traj, nx=NX_DS,\n",
    "        beta=0.25, gamma=0.5, bc='simply_supported',\n",
    "    )\n",
    "\n",
    "    try:\n",
    "        times_ds, U_ds, V_ds, A_ds, stretch_ds = solver_ds.solve(\n",
    "            T=T_DS, dt=DT_DS,\n",
    "            u0_fun=u0, v0_fun=v0,\n",
    "            q_fun=None,\n",
    "            store_every=STORE_EVERY,\n",
    "            tol=1e-8, max_iter=30, verbose=False,\n",
    "        )\n",
    "        dataset_U.append(U_ds)\n",
    "        dataset_stretch.append(stretch_ds)\n",
    "        dataset_params.append((alpha_traj, ic_choice, n_bumps))\n",
    "\n",
    "        if (traj + 1) % 20 == 0:\n",
    "            print(f'  Trajectory {traj+1}/{N_TRAJ} done  '\n",
    "                  f'(alpha={alpha_traj:.1f}, IC={ic_choice}, '\n",
    "                  f'max|w|={np.max(np.abs(U_ds)):.4f}, '\n",
    "                  f'max S={np.max(stretch_ds):.4f})')\n",
    "    except Exception as e:\n",
    "        print(f'  Trajectory {traj+1} FAILED (alpha={alpha_traj:.1f}, IC={ic_choice}): {e}')\n",
    "\n",
    "dataset_U = np.array(dataset_U)\n",
    "print(f'\\nDataset shape: {dataset_U.shape}  '\n",
    "      f'({dataset_U.shape[0]} trajectories x {dataset_U.shape[1]} time steps x {dataset_U.shape[2]} spatial points)')\n",
])

# ---- Cell 5: Markdown for quality check ----
cell_md_quality = make_md_cell([
    "### Quality check: nonlinearity verification\n",
    "\n",
    "Compare a few dataset trajectories against linear solutions to confirm nonlinear effects are present.\n",
])

# ---- Cell 6: Quality check plot ----
cell_quality_check = make_code_cell([
    "# ---------------------------------------------------------------------------\n",
    "# Verify nonlinearity: compare 4 random trajectories against linear solutions\n",
    "# ---------------------------------------------------------------------------\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)\n",
    "check_idx = np.random.choice(len(dataset_U), size=4, replace=False)\n",
    "\n",
    "for ax, ci in zip(axes.flat, check_idx):\n",
    "    alpha_ci, ic_ci, nb_ci = dataset_params[ci]\n",
    "\n",
    "    # Re-solve with alpha=0 (linear) using same IC stored in dataset\n",
    "    # Use first snapshot as IC\n",
    "    u0_snap = dataset_U[ci, 0, :]\n",
    "    lin_solver = NonlinearBeam1D(\n",
    "        L=L_DS, alpha=0.0, nx=NX_DS,\n",
    "        beta=0.25, gamma=0.5, bc='simply_supported',\n",
    "    )\n",
    "    _, U_lin_check, _, _, _ = lin_solver.solve(\n",
    "        T=T_DS, dt=DT_DS,\n",
    "        u0_fun=lambda x, u0s=u0_snap: u0s,\n",
    "        v0_fun=v0_fun_zero,\n",
    "        q_fun=None, store_every=STORE_EVERY,\n",
    "    )\n",
    "\n",
    "    mid = NX_DS // 2\n",
    "    ax.plot(times_ds, dataset_U[ci, :, mid], label='nonlinear')\n",
    "    ax.plot(times_ds, U_lin_check[:, mid], '--', label='linear')\n",
    "    ax.set_title(f'Traj {ci}: α={alpha_ci:.0f}, IC={ic_ci}')\n",
    "    ax.legend(fontsize=8)\n",
    "    ax.set_ylabel('midspan w')\n",
    "\n",
    "axes[1, 0].set_xlabel('time')\n",
    "axes[1, 1].set_xlabel('time')\n",
    "fig.suptitle('Nonlinear vs linear comparison (dataset samples)', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
])

# ---- Cell 7: Markdown for saving ----
cell_md_save = make_md_cell([
    "### Save the nonlinear beam dataset\n",
])

# ---- Cell 8: Save dataset ----
cell_save = make_code_cell([
    "# ---------------------------------------------------------------------------\n",
    "# Save dataset to .npz\n",
    "# ---------------------------------------------------------------------------\n",
    "\n",
    "alphas = np.array([p[0] for p in dataset_params])\n",
    "ic_labels = np.array([p[1] for p in dataset_params])\n",
    "n_bumps_arr = np.array([p[2] for p in dataset_params])\n",
    "stretch_arr = np.array(dataset_stretch)\n",
    "\n",
    "save_path = 'nonlinear_beam_dataset.npz'\n",
    "\n",
    "np.savez_compressed(\n",
    "    save_path,\n",
    "    U=dataset_U,\n",
    "    times=times_ds,\n",
    "    x=np.linspace(0, L_DS, NX_DS),\n",
    "    stretch=stretch_arr,\n",
    "    alpha=alphas,\n",
    "    ic_type=ic_labels,\n",
    "    n_bumps=n_bumps_arr,\n",
    "    L=L_DS,\n",
    "    nx=NX_DS,\n",
    "    dt=DT_DS,\n",
    "    T=T_DS,\n",
    "    store_every=STORE_EVERY,\n",
    ")\n",
    "\n",
    "print(f'Dataset saved to {save_path}')\n",
    "print(f'  U shape:       {dataset_U.shape}')\n",
    "print(f'  times shape:   {times_ds.shape}')\n",
    "print(f'  alpha range:   [{alphas.min():.1f}, {alphas.max():.1f}]')\n",
    "print(f'  stretch range: [{stretch_arr.min():.6f}, {stretch_arr.max():.4f}]')\n",
    "print(f'  IC types:      {dict(zip(*np.unique(ic_labels, return_counts=True)))}')\n",
])

# Append all new cells
new_cells = [
    cell_md_header,
    cell_ic_generators,
    cell_md_loop,
    cell_dataset_loop,
    cell_md_quality,
    cell_quality_check,
    cell_md_save,
    cell_save,
]

nb['cells'].extend(new_cells)

with open('notebooks/nonlinear_beam_solver_1d_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(new_cells)} cells. Total cells: {len(nb['cells'])}")
