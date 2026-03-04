"""Data preparation helpers for VIV CDE/SHRED workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd

from utils.models_diffrax import prepare_data_CDE

import jax.numpy as jnp

def mse(datapred, datatrue):
    return jnp.mean(jnp.sum((datapred - datatrue)**2, axis = -1))
def rmsre(datapred, datatrue):
    return jnp.mean(jnp.sqrt(jnp.sum((datapred - datatrue)**2, axis = -1))/jnp.sqrt(jnp.sum((datatrue)**2, axis = -1)))

def mae(datapred, datatrue):
    return jnp.mean(jnp.sum(jnp.abs(datapred - datatrue), axis = -1))

def mre(datapred, datatrue):
    return jnp.mean(jnp.abs(datapred - datatrue)) / jnp.mean(jnp.abs(datatrue))


def load_csv_pairs(
    data_dir: Path | str,
    cf_suffix: str = "DISPLCF.csv",
    il_suffix: str = "DISPLIL.csv",
) -> List[Tuple[Path, Path]]:
    """Find matching CF/IL CSV file pairs in a directory."""
    data_dir = Path(data_dir)

    def strip_suffix(name: str, suffix: str) -> str | None:
        if name.endswith(suffix):
            return name[: -len(suffix)]
        return None

    cf_files: Dict[str, Path] = {}
    il_files: Dict[str, Path] = {}

    for path in data_dir.glob(f"*{cf_suffix}"):
        base = strip_suffix(path.name, cf_suffix)
        if base is not None:
            cf_files[base] = path

    for path in data_dir.glob(f"*{il_suffix}"):
        base = strip_suffix(path.name, il_suffix)
        if base is not None:
            il_files[base] = path

    pairs = []
    for base in sorted(set(cf_files) & set(il_files)):
        pairs.append((cf_files[base], il_files[base]))

    return pairs


def extract_dataframe(
    path_cf: Path | str,
    path_il: Path | str,
    stride: int | None = 2,
    transpose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CF/IL CSVs and return time-first arrays."""
    cf = pd.read_csv(path_cf).to_numpy()
    il = pd.read_csv(path_il).to_numpy()

    if stride is not None:
        cf = cf[:, ::stride]
        il = il[:, ::stride]

    if transpose:
        cf = cf.T
        il = il.T

    min_len = min(cf.shape[0], il.shape[0])
    return cf[:min_len], il[:min_len]


def create_sequences(data: np.ndarray, lag_size: int, stride: int = 1, add_time: bool = True) -> np.ndarray:
    """Create lag-windowed sequences with optional time coordinate and striding.
    
    Args:
        data: Input data array of shape (n_timesteps, n_features)
        lag_size: Number of timesteps in each sequence
        stride: Spacing between timesteps within each sequence (default: 1).
                stride > 1 creates longer temporal windows with fewer samples.
                Each sequence spans lag_size * stride timesteps.
        add_time: Whether to append normalized time coordinate to the data.
    
    Returns:
        Stacked sequences of shape (n_sequences, lag_size, n_features+add_time)
    """
    if add_time:
        time_col = np.linspace(0, 1, data.shape[0])[:, None]
        data = np.hstack([data, time_col])

    # With stride, each sequence spans lag_size * stride timesteps
    max_start_idx = data.shape[0] - lag_size * stride
    return np.stack([data[i : i + lag_size * stride : stride] for i in range(max_start_idx + 1)])


def split_indices(
    n_total: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create shuffled train/valid/test indices (random across all samples)."""
    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(n_total)
    n_train = int(train_ratio * n_total)
    n_valid = int(valid_ratio * n_total)

    return {
        "train": all_indices[:n_train],
        "valid": all_indices[n_train : n_train + n_valid],
        "test": all_indices[n_train + n_valid :],
    }


def split_indices_by_trajectory(
    n_total: int,
    n_trajectories: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    shuffle_within_split: bool = True,
) -> Dict[str, np.ndarray]:
    """Split indices by trajectory to prevent temporal data leakage.

    The data is conceptually divided into ``n_trajectories`` contiguous chunks
    (trajectories).  Trajectories are assigned **sequentially** to
    train / valid / test so that each split covers a distinct time domain:

        train  = first 80 % of trajectories
        valid  = next  10 %
        test   = last  10 %

    Within each split the sample indices are optionally shuffled so that
    mini-batch training still sees a random order.

    Args:
        n_total: Total number of samples (e.g. number of lag-windows).
        n_trajectories: Number of trajectories to divide the data into.
            ``n_total`` is truncated to ``n_trajectories * traj_len`` so
            that every trajectory has the same length.
        train_ratio: Fraction of trajectories for training.
        valid_ratio: Fraction of trajectories for validation.
        seed: Random seed used only for the within-split shuffle.
        shuffle_within_split: If True, shuffle indices inside each split.

    Returns:
        Dictionary with 'train', 'valid', 'test' index arrays and metadata.
    """
    traj_len = n_total // n_trajectories
    n_usable = traj_len * n_trajectories  # drop remainder samples

    n_train_traj = max(1, int(train_ratio * n_trajectories))
    n_valid_traj = max(1, int(valid_ratio * n_trajectories))
    n_test_traj = n_trajectories - n_train_traj - n_valid_traj
    assert n_test_traj >= 1, (
        f"Not enough trajectories for test split: "
        f"{n_trajectories} total, {n_train_traj} train, {n_valid_traj} valid"
    )

    all_indices = np.arange(n_usable)

    # Sequential trajectory assignment
    train_end = n_train_traj * traj_len
    valid_end = train_end + n_valid_traj * traj_len

    train_idx = all_indices[:train_end]
    valid_idx = all_indices[train_end:valid_end]
    test_idx = all_indices[valid_end:n_usable]

    # Optionally shuffle within each split
    if shuffle_within_split:
        rng = np.random.default_rng(seed)
        rng.shuffle(train_idx)
        rng.shuffle(valid_idx)
        rng.shuffle(test_idx)

    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
        "traj_len": traj_len,
        "n_trajectories": n_trajectories,
        "n_train_traj": n_train_traj,
        "n_valid_traj": n_valid_traj,
        "n_test_traj": n_test_traj,
    }


def split_data(
    X: np.ndarray, Y: np.ndarray, indices_dict: Dict[str, np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {
        k: (X[idx], Y[idx])
        for k, idx in indices_dict.items()
        if isinstance(idx, np.ndarray)
    }


def compute_pod_basis(
    Y_train: np.ndarray,
    n_modes: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute POD basis for training snapshots."""
    Y_2d = Y_train.reshape(-1, Y_train.shape[-1])
    _, S_full, Vt = randomized_svd(Y_2d, n_components=n_modes, random_state=random_state)
    return Vt[:n_modes, :], S_full[:n_modes]


def _report_pod_error(
    Y_train: np.ndarray, V: np.ndarray, label: str,
) -> float:
    """Print POD reconstruction error (relative Frobenius norm) and return it."""
    Y_2d = Y_train.reshape(-1, Y_train.shape[-1])
    Y_rec = (Y_2d @ V.T) @ V
    rel_err = np.linalg.norm(Y_2d - Y_rec) / np.linalg.norm(Y_2d)
    pct = rel_err * 100
    n_modes = V.shape[0]
    print(f"  POD {label}: {n_modes} modes → relative reconstruction error = {pct:.4f}%")
    return rel_err


def project_to_pod(Y: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.dot(Y, V.T)


def project_split_to_pod(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]], V: np.ndarray
) -> Dict[str, np.ndarray]:
    return {k: project_to_pod(Y, V) for k, (_, Y) in splits.items()}


def prepare_viv_datasets(
    cf_data: np.ndarray,
    il_data: np.ndarray,
    lag: int = 25,
    nsensors: int = 3,
    modes_cf: int = 10,
    modes_il: int = 15,
    seed: int = 42,
    n_trajectories: int | None = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Dict[str, object]:
    """Prepare datasets for Neural CDE and SHRED workflows.

    Args:
        cf_data: Cross-flow data array of shape (n_timesteps, n_spatial).
        il_data: In-line data array of shape (n_timesteps, n_spatial).
        lag: Window / lag size for creating sequences.
        nsensors: Number of sensor locations to sample.
        modes_cf: Number of POD modes for cross-flow.
        modes_il: Number of POD modes for in-line.
        seed: Random seed.
        n_trajectories: If provided, split data into this many contiguous
            trajectories and assign them sequentially to train/valid/test
            (80/10/10 by default).  This prevents temporal data leakage
            that occurs with the default random-index split.
            If ``None``, the legacy random split is used.
        train_ratio: Fraction of trajectories (or samples) for training.
        valid_ratio: Fraction for validation. Test gets the remainder.
    """
    min_len = min(cf_data.shape[0], il_data.shape[0])
    cf_data = cf_data[:min_len]
    il_data = il_data[:min_len]

    nt, nx = cf_data.shape
    rng = np.random.default_rng(seed)
    sensor_spatial = rng.choice(np.arange(0, nx - 1), size=nsensors, replace=False).tolist()
    sensor_locations = sensor_spatial + [-1]

    seq_cf = create_sequences(cf_data, lag, add_time=True)
    seq_il = create_sequences(il_data, lag, add_time=True)

    X_cf = seq_cf[:, :, sensor_locations]
    Y_cf = seq_cf[:, -1, :-1]

    X_il = seq_il[:, :, sensor_locations]
    Y_il = seq_il[:, -1, :-1]

    indices = (
        split_indices_by_trajectory(
            X_cf.shape[0],
            n_trajectories=n_trajectories,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
        if n_trajectories is not None
        else split_indices(
            X_cf.shape[0],
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    )
    cf_splits = split_data(X_cf, Y_cf, indices)
    il_splits = split_data(X_il, Y_il, indices)

    # Print split information
    if n_trajectories is not None:
        traj_len = indices["traj_len"]
        print(f"Trajectory-based split: {n_trajectories} trajectories of {traj_len} timesteps each")
        print(f"  Train: {indices['n_train_traj']} trajectories ({len(indices['train'])} samples)")
        print(f"  Valid: {indices['n_valid_traj']} trajectories ({len(indices['valid'])} samples)")
        print(f"  Test:  {indices['n_test_traj']} trajectories ({len(indices['test'])} samples)")
        dropped = X_cf.shape[0] - traj_len * n_trajectories
        if dropped > 0:
            print(f"  Dropped {dropped} remainder samples to get equal-length trajectories")
    else:
        print(f"Random split (legacy): train={len(indices['train'])}, "
              f"valid={len(indices['valid'])}, test={len(indices['test'])}")

    V_cf, S_cf = compute_pod_basis(cf_splits["train"][1], modes_cf, random_state=seed)
    V_il, S_il = compute_pod_basis(il_splits["train"][1], modes_il, random_state=seed)
    _report_pod_error(cf_splits["train"][1], V_cf, "CF")
    _report_pod_error(il_splits["train"][1], V_il, "IL")

    cf_pod = project_split_to_pod(cf_splits, V_cf)
    il_pod = project_split_to_pod(il_splits, V_il)

    pod_train = np.concatenate([cf_pod["train"], il_pod["train"]], axis=-1)
    pod_valid = np.concatenate([cf_pod["valid"], il_pod["valid"]], axis=-1)
    pod_test = np.concatenate([cf_pod["test"], il_pod["test"]], axis=-1)

    scaler = MinMaxScaler().fit(pod_train)
    pod_train_scaled = scaler.transform(pod_train)
    pod_valid_scaled = scaler.transform(pod_valid)
    pod_test_scaled = scaler.transform(pod_test)

    X_train = np.concatenate([cf_splits["train"][0][:, :, :-1], il_splits["train"][0]], axis=-1)
    X_valid = np.concatenate([cf_splits["valid"][0][:, :, :-1], il_splits["valid"][0]], axis=-1)
    X_test = np.concatenate([cf_splits["test"][0][:, :, :-1], il_splits["test"][0]], axis=-1)

    train_data_cde, _ = prepare_data_CDE(X_train, pod_train_scaled)
    valid_data_cde, _ = prepare_data_CDE(X_valid, pod_valid_scaled)
    test_data_cde, _ = prepare_data_CDE(X_test, pod_test_scaled)

    X_combined = np.concatenate([X_cf[:, :, :-1], X_il], axis=-1)
    pod_combined = np.concatenate([np.dot(Y_cf, V_cf.T), np.dot(Y_il, V_il.T)], axis=-1)
    pod_combined_scaled = scaler.transform(pod_combined)
    full_data_cde, _ = prepare_data_CDE(X_combined, pod_combined_scaled)

    cde_splits = {
        "train": {
            "data": train_data_cde,
            "sensors_with_time": X_train,
            "sensors_raw": X_train[:, :, :-1],
            "pod_scaled": pod_train_scaled,
            "pod_unscaled": pod_train,
        },
        "valid": {
            "data": valid_data_cde,
            "sensors_with_time": X_valid,
            "sensors_raw": X_valid[:, :, :-1],
            "pod_scaled": pod_valid_scaled,
            "pod_unscaled": pod_valid,
        },
        "test": {
            "data": test_data_cde,
            "sensors_with_time": X_test,
            "sensors_raw": X_test[:, :, :-1],
            "pod_scaled": pod_test_scaled,
            "pod_unscaled": pod_test,
        },
        "sequential": {
            "data": full_data_cde,
            "sensors_with_time": X_combined,
            "sensors_raw": X_combined[:, :, :-1],
            "pod_scaled": pod_combined_scaled,
            "pod_unscaled": pod_combined,
        },
    }

    shred_splits = {
        "train": {
            "S_i": X_train[:, :, :-1],
            "Y": pod_train_scaled,
            "pod_scaled": pod_train_scaled,
            "pod_unscaled": pod_train,
            "sensors_raw": X_train[:, :, :-1],
        },
        "valid": {
            "S_i": X_valid[:, :, :-1],
            "Y": pod_valid_scaled,
            "pod_scaled": pod_valid_scaled,
            "pod_unscaled": pod_valid,
            "sensors_raw": X_valid[:, :, :-1],
        },
        "test": {
            "S_i": X_test[:, :, :-1],
            "Y": pod_test_scaled,
            "pod_scaled": pod_test_scaled,
            "pod_unscaled": pod_test,
            "sensors_raw": X_test[:, :, :-1],
        },
        "sequential": {
            "S_i": X_combined[:, :, :-1],
            "Y": pod_combined_scaled,
            "pod_scaled": pod_combined_scaled,
            "pod_unscaled": pod_combined,
            "sensors_raw": X_combined[:, :, :-1],
        },
    }

    split_info = {
        "method": "trajectory" if n_trajectories is not None else "random",
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "n_train": len(indices["train"]),
        "n_valid": len(indices["valid"]),
        "n_test": len(indices["test"]),
    }
    if n_trajectories is not None:
        split_info.update({
            "n_trajectories": n_trajectories,
            "traj_len": indices["traj_len"],
            "n_train_traj": indices["n_train_traj"],
            "n_valid_traj": indices["n_valid_traj"],
            "n_test_traj": indices["n_test_traj"],
        })

    return {
        "raw": {"cf": cf_data, "il": il_data},
        "sensor_locations": sensor_locations,
        "sensor_spatial": sensor_spatial,
        "indices": indices,
        "split_info": split_info,
        "pod": {
            "cf": {"V": V_cf, "S": S_cf},
            "il": {"V": V_il, "S": S_il},
        },
        "scaler": scaler,
        "cde": cde_splits,
        "shred": shred_splits,
        "full_fields": {"cf": Y_cf, "il": Y_il},
    }


def prepare_viv_datasets_from_csv(
    cf_path: Path | str,
    il_path: Path | str,
    lag: int = 25,
    nsensors: int = 3,
    modes_cf: int = 10,
    modes_il: int = 15,
    seed: int = 42,
    stride: int | None = 2,
    transpose: bool = True,
    n_trajectories: int | None = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Dict[str, object]:
    """Load CSVs and prepare datasets for Neural CDE and SHRED."""
    cf_data, il_data = extract_dataframe(cf_path, il_path, stride=stride, transpose=transpose)
    return prepare_viv_datasets(
        cf_data,
        il_data,
        lag=lag,
        nsensors=nsensors,
        modes_cf=modes_cf,
        modes_il=modes_il,
        seed=seed,
        n_trajectories=n_trajectories,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )


def prepare_stacked_viv_datasets(
    pairs: List[Tuple[Path | str, Path | str]],
    lag: int = 25,
    nsensors: int = 3,
    modes_cf: int = 10,
    modes_il: int = 15,
    seed: int = 42,
    stride: int | None = 2,
    transpose: bool = True,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Dict[str, object]:
    """Stack multiple CF/IL pairs and return the same dict as ``prepare_viv_datasets``.

    Each pair corresponds to a different riser speed. The pairs are
    individually windowed into lag-sequences and then concatenated. A shared
    POD basis and scaler are computed from the training split. The returned
    dictionary has exactly the same structure as :func:`prepare_viv_datasets`
    so downstream code can treat stacked and single-pair data identically.

    Args:
        pairs: List of (cf_path, il_path) tuples to stack.
        lag: Window / lag size for creating sequences.
        nsensors: Number of sensor locations to sample.
        modes_cf: Number of POD modes for cross-flow.
        modes_il: Number of POD modes for in-line.
        seed: Random seed.
        stride: Spatial stride when loading CSVs.
        transpose: Whether to transpose CSVs (time-first).
        train_ratio: Train fraction *within* the stacked data.
        valid_ratio: Validation fraction *within* the stacked data.

    Returns:
        Dictionary with the same structure as ``prepare_viv_datasets`` plus
        extra metadata about which pairs were stacked.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1) Load and window each pair independently
    # ------------------------------------------------------------------
    all_X_cf, all_Y_cf = [], []
    all_X_il, all_Y_il = [], []
    pair_boundaries = []          # (start_idx, end_idx) per pair
    pair_names: list[str] = []
    sensor_locations = None

    offset = 0
    for cf_path, il_path in pairs:
        cf_data, il_data = extract_dataframe(cf_path, il_path, stride=stride, transpose=transpose)
        min_len = min(cf_data.shape[0], il_data.shape[0])
        cf_data, il_data = cf_data[:min_len], il_data[:min_len]
        nt, nx = cf_data.shape

        # Sensor locations – fixed across all pairs for consistency
        if sensor_locations is None:
            sensor_spatial = rng.choice(np.arange(0, nx - 1), size=nsensors, replace=False).tolist()
            sensor_locations = sensor_spatial + [-1]

        seq_cf = create_sequences(cf_data, lag, add_time=True)
        seq_il = create_sequences(il_data, lag, add_time=True)

        X_cf = seq_cf[:, :, sensor_locations]
        Y_cf = seq_cf[:, -1, :-1]
        X_il = seq_il[:, :, sensor_locations]
        Y_il = seq_il[:, -1, :-1]

        n_windows = X_cf.shape[0]
        pair_boundaries.append((offset, offset + n_windows))
        pair_names.append(f"{Path(cf_path).stem} + {Path(il_path).stem}")
        offset += n_windows

        all_X_cf.append(X_cf);  all_Y_cf.append(Y_cf)
        all_X_il.append(X_il);  all_Y_il.append(Y_il)

        print(f"  Pair {len(pair_names)}: {Path(cf_path).stem}  →  {n_windows} windows  (nt={nt}, nx={nx})")

    X_cf = np.concatenate(all_X_cf, axis=0)
    Y_cf = np.concatenate(all_Y_cf, axis=0)
    X_il = np.concatenate(all_X_il, axis=0)
    Y_il = np.concatenate(all_Y_il, axis=0)
    print(f"Stacked pool: {X_cf.shape[0]} total windows from {len(pairs)} pairs")

    # ------------------------------------------------------------------
    # 2) Train / valid / test split on the stacked pool
    # ------------------------------------------------------------------
    indices = split_indices(
        X_cf.shape[0],
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    cf_splits = split_data(X_cf, Y_cf, indices)
    il_splits = split_data(X_il, Y_il, indices)

    n_train = len(indices["train"])
    n_valid = len(indices["valid"])
    n_test = len(indices["test"])
    print(f"  Split → train={n_train}, valid={n_valid}, test={n_test}")

    # ------------------------------------------------------------------
    # 3) POD basis & scaler from training data
    # ------------------------------------------------------------------
    V_cf, S_cf = compute_pod_basis(cf_splits["train"][1], modes_cf, random_state=seed)
    V_il, S_il = compute_pod_basis(il_splits["train"][1], modes_il, random_state=seed)
    _report_pod_error(cf_splits["train"][1], V_cf, "CF")
    _report_pod_error(il_splits["train"][1], V_il, "IL")

    cf_pod = project_split_to_pod(cf_splits, V_cf)
    il_pod = project_split_to_pod(il_splits, V_il)

    pod_train = np.concatenate([cf_pod["train"], il_pod["train"]], axis=-1)
    pod_valid = np.concatenate([cf_pod["valid"], il_pod["valid"]], axis=-1)
    pod_test = np.concatenate([cf_pod["test"], il_pod["test"]], axis=-1)

    scaler = MinMaxScaler().fit(pod_train)
    pod_train_scaled = scaler.transform(pod_train)
    pod_valid_scaled = scaler.transform(pod_valid)
    pod_test_scaled = scaler.transform(pod_test)

    # Sensor inputs (drop time col from cf, keep all from il which includes time)
    X_train = np.concatenate([cf_splits["train"][0][:, :, :-1], il_splits["train"][0]], axis=-1)
    X_valid = np.concatenate([cf_splits["valid"][0][:, :, :-1], il_splits["valid"][0]], axis=-1)
    X_test = np.concatenate([cf_splits["test"][0][:, :, :-1], il_splits["test"][0]], axis=-1)

    train_data_cde, _ = prepare_data_CDE(X_train, pod_train_scaled)
    valid_data_cde, _ = prepare_data_CDE(X_valid, pod_valid_scaled)
    test_data_cde, _ = prepare_data_CDE(X_test, pod_test_scaled)

    # ------------------------------------------------------------------
    # 4) Build sequential view over full stacked data
    # ------------------------------------------------------------------
    X_combined = np.concatenate([X_cf[:, :, :-1], X_il], axis=-1)
    pod_combined = np.concatenate(
        [np.dot(Y_cf, V_cf.T), np.dot(Y_il, V_il.T)], axis=-1
    )
    pod_combined_scaled = scaler.transform(pod_combined)
    full_data_cde, _ = prepare_data_CDE(X_combined, pod_combined_scaled)

    # ------------------------------------------------------------------
    # 5) Assemble output dicts (same structure as prepare_viv_datasets)
    # ------------------------------------------------------------------
    cde_splits = {
        "train": {
            "data": train_data_cde,
            "sensors_with_time": X_train,
            "sensors_raw": X_train[:, :, :-1],
            "pod_scaled": pod_train_scaled,
            "pod_unscaled": pod_train,
        },
        "valid": {
            "data": valid_data_cde,
            "sensors_with_time": X_valid,
            "sensors_raw": X_valid[:, :, :-1],
            "pod_scaled": pod_valid_scaled,
            "pod_unscaled": pod_valid,
        },
        "test": {
            "data": test_data_cde,
            "sensors_with_time": X_test,
            "sensors_raw": X_test[:, :, :-1],
            "pod_scaled": pod_test_scaled,
            "pod_unscaled": pod_test,
        },
        "sequential": {
            "data": full_data_cde,
            "sensors_with_time": X_combined,
            "sensors_raw": X_combined[:, :, :-1],
            "pod_scaled": pod_combined_scaled,
            "pod_unscaled": pod_combined,
        },
    }

    shred_splits = {
        "train": {
            "S_i": X_train[:, :, :-1],
            "Y": pod_train_scaled,
            "pod_scaled": pod_train_scaled,
            "pod_unscaled": pod_train,
            "sensors_raw": X_train[:, :, :-1],
        },
        "valid": {
            "S_i": X_valid[:, :, :-1],
            "Y": pod_valid_scaled,
            "pod_scaled": pod_valid_scaled,
            "pod_unscaled": pod_valid,
            "sensors_raw": X_valid[:, :, :-1],
        },
        "test": {
            "S_i": X_test[:, :, :-1],
            "Y": pod_test_scaled,
            "pod_scaled": pod_test_scaled,
            "pod_unscaled": pod_test,
            "sensors_raw": X_test[:, :, :-1],
        },
        "sequential": {
            "S_i": X_combined[:, :, :-1],
            "Y": pod_combined_scaled,
            "pod_scaled": pod_combined_scaled,
            "pod_unscaled": pod_combined,
            "sensors_raw": X_combined[:, :, :-1],
        },
    }

    stacking_info = {
        "pairs": pair_names,
        "pair_boundaries": pair_boundaries,
        "n_train_windows": n_train,
        "n_valid_windows": n_valid,
        "n_test_windows": n_test,
    }

    return {
        "raw": {"cf": Y_cf, "il": Y_il},
        "sensor_locations": sensor_locations,
        "sensor_spatial": sensor_spatial,
        "indices": indices,
        "stacking_info": stacking_info,
        "pod": {
            "cf": {"V": V_cf, "S": S_cf},
            "il": {"V": V_il, "S": S_il},
        },
        "scaler": scaler,
        "cde": cde_splits,
        "shred": shred_splits,
        "full_fields": {"cf": Y_cf, "il": Y_il},
    }
