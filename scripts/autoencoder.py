"""
autoencoder.py
==============
Trains a nonlinear MLP autoencoder on 1-D transport equation snapshots.

Goals
-----
1. Find the minimum latent dimension that achieves target reconstruction quality.
2. Compare nonlinear (autoencoder) vs. linear (POD) compression.
3. Suggest a concrete decoder MLP architecture for use in SHRED.

Usage
-----
    python scripts/autoencoder.py [--latent_dims 4 8 16 32] [--epochs 300]

Output
------
  - Reconstruction MRE table (autoencoder vs POD) for each latent dim
  - Loss curves saved to figures/autoencoder_loss.png
  - Suggested decoder architecture printed to stdout
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ── Project root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))

# ── Simulation parameters (must match notebook) ───────────────────────────
L      = 2 * np.pi
Nx     = 128
Nt     = 400
T      = 12.0
dt     = T / Nt
x      = np.linspace(0, L, Nx, endpoint=False)
t_grid = np.linspace(0, T, Nt)

n_train = 100
n_test  =  20
n_total = n_train + n_test


# ─────────────────────────────────────────────────────────────────────────────
# Data generation  (identical to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def solve_transport(u0: np.ndarray, c: float) -> np.ndarray:
    L_ = x[-1] + (x[1] - x[0])
    snaps = np.empty((Nt, Nx), dtype=np.float32)
    for ti, tt in enumerate(t_grid):
        x_shift = (x - c * tt) % L_
        snaps[ti] = np.interp(x_shift, x, u0, period=L_)
    return snaps


rng = np.random.default_rng(42)


def ic_gaussian_bumps(n_bumps=None):
    L_ = x[-1] + (x[1] - x[0])
    if n_bumps is None:
        n_bumps = rng.integers(1, 4)
    u = np.zeros(Nx, dtype=np.float32)
    for _ in range(n_bumps):
        x0    = rng.uniform(0, L_)
        sigma = rng.uniform(0.2, 0.7)
        amp   = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
        for shift in [-1, 0, 1]:
            u += amp * np.exp(-0.5 * ((x - x0 + shift * L_) / sigma) ** 2)
    return u.astype(np.float32)


def ic_fourier_modes(Kmax=8):
    L_ = x[-1] + (x[1] - x[0])
    u  = np.zeros(Nx, dtype=np.float32)
    for k in range(1, Kmax + 1):
        ak = rng.normal(0, 1.0 / k ** 1.5)
        bk = rng.normal(0, 1.0 / k ** 1.5)
        u += ak * np.cos(2 * np.pi * k * x / L_)
        u += bk * np.sin(2 * np.pi * k * x / L_)
    return u.astype(np.float32)


def ic_mixed():
    alpha = rng.uniform(0.3, 0.7)
    return (alpha * ic_gaussian_bumps() + (1 - alpha) * ic_fourier_modes()).astype(np.float32)


def generate_dataset():
    speeds = np.concatenate([
        rng.uniform(0.5, 3.0, n_total // 2),
        rng.uniform(-3.0, -0.5, n_total - n_total // 2),
    ])
    ic_gens = [ic_gaussian_bumps, ic_fourier_modes, ic_mixed]
    all_snaps = []
    for i in range(n_total):
        u0    = ic_gens[i % len(ic_gens)]()
        snaps = solve_transport(u0, speeds[i])
        all_snaps.append(snaps)
    all_snaps = np.stack(all_snaps)          # (n_total, Nt, Nx)
    snaps_train = all_snaps[:n_train]
    snaps_test  = all_snaps[n_train:]
    return snaps_train, snaps_test


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder architecture
# ─────────────────────────────────────────────────────────────────────────────

def make_encoder(nx: int, latent_dim: int, hidden: list[int]) -> nn.Sequential:
    layers = []
    in_dim = nx
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.GELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, latent_dim))
    return nn.Sequential(*layers)


def make_decoder(latent_dim: int, nx: int, hidden: list[int]) -> nn.Sequential:
    layers = []
    in_dim = latent_dim
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.GELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, nx))
    return nn.Sequential(*layers)


class MLP_Autoencoder(nn.Module):
    def __init__(self, nx: int, latent_dim: int, hidden: list[int]):
        super().__init__()
        self.encoder = make_encoder(nx, latent_dim, hidden)
        self.decoder = make_decoder(latent_dim, nx, list(reversed(hidden)))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
# POD baseline
# ─────────────────────────────────────────────────────────────────────────────

def pod_reconstruct(data_flat: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """SVD-based POD reconstruction.

    data_flat : (N, Nx)
    Returns   : reconstruction (N, Nx), modes (Nx, r)
    """
    U, s, Vt = np.linalg.svd(data_flat - data_flat.mean(0, keepdims=True), full_matrices=False)
    Phi     = Vt[:r].T                         # (Nx, r)
    mean_   = data_flat.mean(0)
    centered = data_flat - mean_
    coords  = centered @ Phi                   # (N, r)
    recon   = coords @ Phi.T + mean_           # (N, Nx)
    return recon, Phi


def mre(yt: np.ndarray, yp: np.ndarray) -> float:
    num = np.sqrt(((yt - yp) ** 2).sum(axis=-1))
    den = np.sqrt((yt ** 2).sum(axis=-1)) + 1e-12
    return float((num / den).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    train_flat: np.ndarray,
    val_flat:   np.ndarray,
    latent_dim: int,
    hidden:     list[int],
    epochs:     int,
    batch_size: int,
    lr:         float,
    device:     torch.device,
) -> tuple[MLP_Autoencoder, list[float], list[float]]:

    # Normalise to zero mean / unit std per feature
    mean_ = train_flat.mean(0, keepdims=True)
    std_  = train_flat.std(0,  keepdims=True) + 1e-8
    X_tr  = torch.tensor((train_flat - mean_) / std_, dtype=torch.float32)
    X_vl  = torch.tensor((val_flat   - mean_) / std_, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr), batch_size=batch_size, shuffle=True)

    model = MLP_Autoencoder(train_flat.shape[1], latent_dim, hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_losses, vl_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        batch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = nn.functional.mse_loss(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            batch_loss += loss.item() * len(xb)
        sched.step()
        tr_losses.append(batch_loss / len(X_tr))

        model.eval()
        with torch.no_grad():
            vl_losses.append(nn.functional.mse_loss(
                model(X_vl.to(device)), X_vl.to(device)
            ).item())

        if ep % 50 == 0 or ep == 1:
            print(f"  [latent={latent_dim:3d}] epoch {ep:4d}/{epochs}  "
                  f"train={tr_losses[-1]:.4e}  val={vl_losses[-1]:.4e}")

    # Store normalisation on model for later reconstruction
    model._mean = mean_
    model._std  = std_
    return model, tr_losses, vl_losses


def reconstruct(model: MLP_Autoencoder, data_flat: np.ndarray, device: torch.device) -> np.ndarray:
    X = torch.tensor((data_flat - model._mean) / model._std, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon_norm = model(X).cpu().numpy()
    return recon_norm * model._std + model._mean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dims", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64])
    parser.add_argument("--hidden",      type=int, nargs="+",
                        default=[256, 128])
    parser.add_argument("--epochs",      type=int, default=300)
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--val_frac",    type=float, default=0.15)
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # ── Generate / load data ──────────────────────────────────────────────
    print("Generating snapshots …")
    snaps_train, snaps_test = generate_dataset()
    print(f"  snaps_train {snaps_train.shape}  snaps_test {snaps_test.shape}")

    train_flat = snaps_train.reshape(-1, Nx)   # (n_train*Nt, Nx)
    test_flat  = snaps_test.reshape(-1, Nx)    # (n_test*Nt,  Nx)

    # Trajectory-level train/val split (no window overlap)
    n_val_traj = max(1, int(round(n_train * args.val_frac)))
    n_tr_traj  = n_train - n_val_traj
    tr_flat    = snaps_train[:n_tr_traj].reshape(-1, Nx)
    vl_flat    = snaps_train[n_tr_traj:].reshape(-1, Nx)
    print(f"  Train: {n_tr_traj} traj ({len(tr_flat)} snaps)  "
          f"Val: {n_val_traj} traj ({len(vl_flat)} snaps)")

    # ── POD baseline ──────────────────────────────────────────────────────
    print("\nPOD baseline (on test set):")
    pod_results = {}
    for r in args.latent_dims:
        recon, _ = pod_reconstruct(
            np.vstack([tr_flat, vl_flat]),   # fit on all train
            r
        )
        # Evaluate on test
        mean_ = tr_flat.mean(0)
        std_  = tr_flat.std(0) + 1e-8
        full_flat = np.vstack([tr_flat, vl_flat])
        _, Phi = pod_reconstruct(full_flat, r)
        centered_test = test_flat - full_flat.mean(0, keepdims=True)
        coords = centered_test @ Phi
        recon_test = coords @ Phi.T + full_flat.mean(0, keepdims=True)
        pod_mre = mre(test_flat, recon_test)
        pod_results[r] = pod_mre
        print(f"  r={r:3d}  test MRE = {pod_mre*100:.3f}%")

    # ── Autoencoder sweep ─────────────────────────────────────────────────
    ae_results   = {}
    all_tr_loss  = {}
    all_vl_loss  = {}
    figures_dir  = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    for latent_dim in args.latent_dims:
        print(f"\n── Autoencoder  latent_dim={latent_dim} "
              f"  hidden={args.hidden} ──")
        model, tr_l, vl_l = train_autoencoder(
            tr_flat, vl_flat, latent_dim, args.hidden,
            args.epochs, args.batch_size, args.lr, device,
        )
        recon_test  = reconstruct(model, test_flat, device)
        test_mre    = mre(test_flat, recon_test)
        ae_results[latent_dim]  = test_mre
        all_tr_loss[latent_dim] = tr_l
        all_vl_loss[latent_dim] = vl_l
        print(f"  → Test MRE = {test_mre*100:.3f}%  "
              f"(POD r={latent_dim}: {pod_results[latent_dim]*100:.3f}%)")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Latent dim':>12}  {'AE MRE %':>10}  {'POD MRE %':>10}  {'AE gain':>8}")
    print("=" * 60)
    for r in args.latent_dims:
        ae_v  = ae_results[r]  * 100
        pod_v = pod_results[r] * 100
        gain  = pod_v - ae_v
        print(f"  {r:>12d}  {ae_v:>10.3f}  {pod_v:>10.3f}  {gain:>+8.3f}")
    print("=" * 60)

    # ── Decoder architecture recommendation ──────────────────────────────
    # Choose smallest latent dim where AE MRE < 1 % (or best otherwise)
    target_mre = 0.01
    candidates = [r for r in args.latent_dims if ae_results[r] < target_mre]
    best_dim   = min(candidates) if candidates else min(ae_results, key=ae_results.get)
    dec_hidden = list(reversed(args.hidden))

    print(f"\n  Recommended decoder architecture for SHRED:")
    print(f"    Input  : {best_dim}  (latent / n_modes)")
    for i, h in enumerate(dec_hidden):
        print(f"    Linear({best_dim if i==0 else dec_hidden[i-1]}, {h})  → GELU")
    print(f"    Linear({dec_hidden[-1]}, {Nx})  → output field")
    print(f"\n  i.e. set n_modes ≈ {best_dim} and use hidden={dec_hidden} in SHRED decoder.")

    # ── Loss curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(args.latent_dims),
                             figsize=(4 * len(args.latent_dims), 4), sharey=True)
    if len(args.latent_dims) == 1:
        axes = [axes]
    for ax, r in zip(axes, args.latent_dims):
        ep = range(1, args.epochs + 1)
        ax.semilogy(ep, all_tr_loss[r], label="train")
        ax.semilogy(ep, all_vl_loss[r], label="val",   ls="--")
        ax.set_title(f"latent={r}\nMRE={ae_results[r]*100:.2f}%", fontsize=9)
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("MSE loss")
        ax.grid(alpha=0.3)
    fig.suptitle("Autoencoder training — loss curves per latent dim", fontsize=11)
    plt.tight_layout()
    out_path = figures_dir / "autoencoder_loss.png"
    fig.savefig(out_path, dpi=130)
    print(f"\n  Loss curves saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
