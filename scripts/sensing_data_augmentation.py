"""Prepare POD-based sensing datasets from a single CSV field file.

This module builds train/valid/test splits for sensing tasks where:
- X: sparse sensor measurements at POD-optimal sensor locations (QDEIM)
- Y: normalized POD coefficients of the full field

The POD basis and normalization are fit on training data only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


def _relative_reconstruction_error(
    snapshots: np.ndarray,
    train_mean: np.ndarray,
    phi: np.ndarray,
) -> float:
    centered = snapshots - train_mean
    coeffs = centered @ phi
    recon = coeffs @ phi.T
    denom = np.linalg.norm(centered)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(centered - recon) / denom)


def _fit_pod_with_threshold(
    train_centered: np.ndarray,
    train_field: np.ndarray,
    valid_field: np.ndarray,
    test_field: np.ndarray,
    train_mean: np.ndarray,
    min_modes: int,
    max_modes: int,
    random_state: int,
    reconstruction_treshold: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    threshold = reconstruction_treshold / 100.0
    best = None

    for modes in range(min_modes, max_modes + 1):
        _, singular_values, vt = randomized_svd(
            train_centered,
            n_components=modes,
            random_state=random_state,
        )
        phi = vt.T.astype(np.float32)

        err_train = _relative_reconstruction_error(train_field, train_mean, phi)
        err_valid = _relative_reconstruction_error(valid_field, train_mean, phi)
        err_test = _relative_reconstruction_error(test_field, train_mean, phi)
        best = (phi, singular_values.astype(np.float32), modes, err_train, err_valid, err_test)

        if max(err_train, err_valid, err_test) <= threshold:
            return best

    assert best is not None
    _, _, modes, err_train, err_valid, err_test = best
    raise ValueError(
        "Unable to satisfy reconstruction_treshold="
        f"{reconstruction_treshold:.4g}% with up to {modes} POD modes. "
        f"Best errors: train={err_train * 100:.4f}%, "
        f"valid={err_valid * 100:.4f}%, test={err_test * 100:.4f}%."
    )


def _validate_split_ratios(train_ratio: float, valid_ratio: float) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in (0, 1).")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0.")


def _load_csv_field(csv_path: str | Path, transpose: bool = True) -> np.ndarray:
    data = pd.read_csv(csv_path).to_numpy(dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CSV data, got shape={data.shape}.")
    if transpose:
        data = data.T
    if data.shape[0] < 3:
        raise ValueError("Not enough timesteps after loading CSV.")
    return data


def _load_and_stack_fields(
    csv_path: str | Path | Sequence[str | Path],
    transpose: bool,
    stride: int,
) -> tuple[list[np.ndarray], list[str]]:
    if isinstance(csv_path, (str, Path)):
        csv_paths = [csv_path]
    else:
        csv_paths = list(csv_path)

    if len(csv_paths) == 0:
        raise ValueError("At least one CSV path must be provided.")

    fields = []
    source_files: list[str] = []
    expected_nx: int | None = None
    for path in csv_paths:
        field = _load_csv_field(csv_path=path, transpose=transpose)
        field = _apply_stride(field, stride=stride)

        if expected_nx is None:
            expected_nx = field.shape[1]
        elif field.shape[1] != expected_nx:
            raise ValueError(
                "All CSV files must have the same spatial dimension. "
                f"Expected nx={expected_nx}, got nx={field.shape[1]} for {path}."
            )

        fields.append(field)
        source_files.append(str(Path(path)))

    return fields, source_files


def _apply_stride(field: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}.")
    return field[::stride]


def _temporal_split(
    field: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_t = field.shape[0]
    n_train = int(train_ratio * n_t)
    n_valid_end = int((train_ratio + valid_ratio) * n_t)

    if n_train < 1 or (n_valid_end - n_train) < 1 or (n_t - n_valid_end) < 1:
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"n_t={n_t}, n_train={n_train}, n_valid={n_valid_end - n_train}, n_test={n_t - n_valid_end}."
        )

    train = field[:n_train]
    valid = field[n_train:n_valid_end]
    test = field[n_valid_end:]
    return train, valid, test


def _truncate_and_reshape_split(
    arr: np.ndarray,
    seq_len: int,
    split_name: str,
) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len}.")

    n_t, n_f = arr.shape
    while seq_len > 0:
        n_full = n_t // seq_len
        n_keep = n_full * seq_len
        if n_keep > 0:
            return arr[:n_keep].reshape(n_full, seq_len, n_f)
        seq_len //= 2

    raise ValueError(
        f"Split '{split_name}' has only {n_t} timesteps, too short to form any sequence."
    )


def _maybe_sequence_splits(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    y_truth_train: np.ndarray,
    y_truth_valid: np.ndarray,
    y_truth_test: np.ndarray,
    seq_len: int | None,
) -> tuple[np.ndarray, ...]:
    if seq_len is None:
        return (
            x_train,
            x_valid,
            x_test,
            y_train,
            y_valid,
            y_test,
            y_truth_train,
            y_truth_valid,
            y_truth_test,
        )

    return (
        _truncate_and_reshape_split(x_train, seq_len=seq_len, split_name="train"),
        _truncate_and_reshape_split(x_valid, seq_len=seq_len, split_name="valid"),
        _truncate_and_reshape_split(x_test, seq_len=seq_len, split_name="test"),
        _truncate_and_reshape_split(y_train, seq_len=seq_len, split_name="train"),
        _truncate_and_reshape_split(y_valid, seq_len=seq_len, split_name="valid"),
        _truncate_and_reshape_split(y_test, seq_len=seq_len, split_name="test"),
        _truncate_and_reshape_split(
            y_truth_train,
            seq_len=seq_len,
            split_name="train",
        ),
        _truncate_and_reshape_split(
            y_truth_valid,
            seq_len=seq_len,
            split_name="valid",
        ),
        _truncate_and_reshape_split(
            y_truth_test,
            seq_len=seq_len,
            split_name="test",
        ),
    )


def _build_split_arrays(
    split_parts: list[np.ndarray],
    split_name: str,
    sensor_idx: np.ndarray,
    train_mean: np.ndarray,
    phi: np.ndarray,
    scaler: StandardScaler,
    seq_len: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    y_truth_parts = []

    for part_idx, part in enumerate(split_parts):
        x = part[:, sensor_idx].astype(np.float32)
        y_unscaled = ((part - train_mean) @ phi).astype(np.float32)
        y = scaler.transform(y_unscaled).astype(np.float32)
        y_truth = part.astype(np.float32)

        if seq_len is not None:
            name = f"{split_name}[{part_idx}]"
            x = _truncate_and_reshape_split(x, seq_len=seq_len, split_name=name)
            y = _truncate_and_reshape_split(y, seq_len=seq_len, split_name=name)
            y_truth = _truncate_and_reshape_split(y_truth, seq_len=seq_len, split_name=name)

        x_parts.append(x)
        y_parts.append(y)
        y_truth_parts.append(y_truth)

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(y_truth_parts, axis=0),
    )


def prepare_sensing_data(
    csv_path: str | Path | Sequence[str | Path],
    train_ratio: float,
    valid_ratio: float,
    n_sensors: int = 5,
    n_modes: int = 12,
    random_state: int = 42,
    transpose: bool = True,
    seq_len: int | None = None,
    stride: int = 1,
    reconstruction_treshold: float | None = None,
    sensor_idx: Sequence[int] | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Build sensing inputs/targets from one or multiple full-field CSV files.

    Args:
        csv_path: Path to one CSV or a list/tuple of CSV paths.
        train_ratio: Fraction for first temporal train split.
        valid_ratio: Fraction for following temporal validation split.
        n_sensors: Number of sensors to select by QDEIM.  Ignored when
            ``sensor_idx`` is provided.
        n_modes: Number of POD modes.
        random_state: Random state for randomized SVD.
        transpose: If True, transpose CSV after reading.
        seq_len: Optional sequence length. If provided, each split is truncated
            to a multiple of seq_len and reshaped to (n_seq, seq_len, features)
            within each source trajectory split before concatenation.
        stride: Temporal downsampling stride (stride=2 keeps every other
            timestep).
        reconstruction_treshold: Maximum allowed reconstruction error percent
            for train/valid/test. If provided, n_modes is increased until all
            splits satisfy this value.
        sensor_idx: Optional explicit sensor indices (0-based, within
            ``[0, Nx)``) to use instead of QDEIM placement.  When supplied,
            ``n_sensors`` is ignored and the provided indices are used directly.

    Returns:
        Dictionary containing train/valid/test X and Y, plus POD/scaler metadata.
    """
    _validate_split_ratios(train_ratio, valid_ratio)

    if reconstruction_treshold is not None and reconstruction_treshold <= 0:
        raise ValueError("reconstruction_treshold must be > 0.")

    fields, source_files = _load_and_stack_fields(
        csv_path=csv_path,
        transpose=transpose,
        stride=stride,
    )

    # Split each trajectory independently, then concatenate each split.
    train_parts = []
    valid_parts = []
    test_parts = []
    for field in fields:
        tr, vl, te = _temporal_split(
            field=field,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        train_parts.append(tr)
        valid_parts.append(vl)
        test_parts.append(te)

    train_field = np.concatenate(train_parts, axis=0).astype(np.float32)
    valid_field = np.concatenate(valid_parts, axis=0).astype(np.float32)
    test_field = np.concatenate(test_parts, axis=0).astype(np.float32)

    n_x = train_field.shape[1]

    if not (1 <= n_sensors <= n_x):
        raise ValueError(f"n_sensors must be in [1, {n_x}], got {n_sensors}.")

    max_modes = min(train_field.shape[0] - 1, n_x)
    if not (1 <= n_modes <= max_modes):
        raise ValueError(f"n_modes must be in [1, {max_modes}], got {n_modes}.")

    train_mean = train_field.mean(axis=0, keepdims=True)
    train_centered = train_field - train_mean

    if reconstruction_treshold is None:
        _, singular_values, vt = randomized_svd(
            train_centered,
            n_components=n_modes,
            random_state=random_state,
        )
        phi = vt.T.astype(np.float32)
        n_modes_selected = n_modes
        recon_err_train = _relative_reconstruction_error(train_field, train_mean, phi)
        recon_err_valid = _relative_reconstruction_error(valid_field, train_mean, phi)
        recon_err_test = _relative_reconstruction_error(test_field, train_mean, phi)
    else:
        (
            phi,
            singular_values,
            n_modes_selected,
            recon_err_train,
            recon_err_valid,
            recon_err_test,
        ) = _fit_pod_with_threshold(
            train_centered=train_centered,
            train_field=train_field,
            valid_field=valid_field,
            test_field=test_field,
            train_mean=train_mean,
            min_modes=n_modes,
            max_modes=max_modes,
            random_state=random_state,
            reconstruction_treshold=reconstruction_treshold,
        )

    # Sensor placement: use caller-supplied indices or fall back to QDEIM.
    if sensor_idx is not None:
        sensor_idx = np.sort(np.asarray(sensor_idx, dtype=np.int32))
        if sensor_idx.ndim != 1 or len(sensor_idx) == 0:
            raise ValueError("sensor_idx must be a non-empty 1-D array of indices.")
        if sensor_idx.min() < 0 or sensor_idx.max() >= n_x:
            raise ValueError(
                f"sensor_idx values must be in [0, {n_x - 1}], "
                f"got min={sensor_idx.min()}, max={sensor_idx.max()}."
            )
    else:
        _, _, pivot_idx = qr(phi @ phi.T, pivoting=True)
        sensor_idx = np.sort(pivot_idx[:n_sensors]).astype(np.int32)

    y_train_unscaled = ((train_field - train_mean) @ phi).astype(np.float32)

    print(f"Selected POD modes: {n_modes_selected}")
    print(f"POD reconstruction error (train): {recon_err_train * 100:.4e} %")
    print(f"POD reconstruction error (valid): {recon_err_valid * 100:.4e} %")
    print(f"POD reconstruction error (test):  {recon_err_test * 100:.4e} %")

    scaler = StandardScaler().fit(y_train_unscaled)

    x_train, y_train, y_truth_train = _build_split_arrays(
        split_parts=train_parts,
        split_name="train",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        seq_len=seq_len,
    )
    x_valid, y_valid, y_truth_valid = _build_split_arrays(
        split_parts=valid_parts,
        split_name="valid",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        seq_len=seq_len,
    )
    x_test, y_test, y_truth_test = _build_split_arrays(
        split_parts=test_parts,
        split_name="test",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        seq_len=seq_len,
    )

    # Build index maps: for each split, record which trajectory indices
    # belong to which source CSV file. Useful for per-riser-speed analysis.
    def _compute_source_index_map(parts, split_name):
        index_map = {}
        cursor = 0
        for i, part in enumerate(parts):
            if seq_len is not None:
                n_seqs = (part.shape[0] // seq_len)
                # mirror the halving logic in _truncate_and_reshape_split
                sl = seq_len
                while sl > 0 and part.shape[0] // sl == 0:
                    sl //= 2
                n_seqs = part.shape[0] // sl if sl > 0 else 0
            else:
                n_seqs = part.shape[0]
            index_map[source_files[i]] = list(range(cursor, cursor + n_seqs))
            cursor += n_seqs
        return index_map

    source_index = {
        "train": _compute_source_index_map(train_parts, "train"),
        "valid": _compute_source_index_map(valid_parts, "valid"),
        "test":  _compute_source_index_map(test_parts, "test"),
    }

    print(
        "X shapes:",
        {
            "train": x_train.shape,
            "valid": x_valid.shape,
            "test": x_test.shape,
        },
    )
    print(
        "Y shapes:",
        {
            "train": y_train.shape,
            "valid": y_valid.shape,
            "test": y_test.shape,
        },
    )

    return {
        "source_files": source_files,
        "n_source_files": len(source_files),
        "sensor_idx": sensor_idx,
        "pod_basis": phi,
        "singular_values": singular_values.astype(np.float32),
        "n_modes_selected": n_modes_selected,
        "train_mean": train_mean.astype(np.float32),
        "stride": stride,
        "seq_len": seq_len,
        "reconstruction_treshold": reconstruction_treshold,
        "reconstruction_error": {
            "train": recon_err_train,
            "valid": recon_err_valid,
            "test": recon_err_test,
        },
        "scaler": scaler,
        "source_index": source_index,
        # Raw split parts kept so sensor extraction can be redone cheaply
        # without re-reading CSVs or re-running POD.  Shape: list of (n_t, n_x).
        "raw_parts": {"train": train_parts, "valid": valid_parts, "test": test_parts},
        "train": {"X": x_train, "Y": y_train, "Y_truth": y_truth_train},
        "valid": {"X": x_valid, "Y": y_valid, "Y_truth": y_truth_valid},
        "test": {"X": x_test, "Y": y_test, "Y_truth": y_truth_test},
    }


def rebuild_sensor_inputs(
    dataset: Dict[str, Any],
    sensor_idx: Sequence[int] | np.ndarray | None = None,
    n_sensors: int | None = None,
) -> Dict[str, Any]:
    """Re-extract sensor inputs from stored raw parts without re-loading CSVs or re-running POD.

    Provide exactly one of ``sensor_idx`` (explicit indices) or ``n_sensors``
    (QDEIM placement on the existing POD basis).  Returns a shallow copy of
    ``dataset`` with updated ``sensor_idx`` and ``train``/``valid``/``test`` X arrays.

    Args:
        dataset:    Dictionary returned by ``prepare_sensing_data``.
        sensor_idx: Explicit 0-based sensor indices to use.
        n_sensors:  Number of QDEIM sensors to select from the existing POD basis.

    Returns:
        Updated dataset dict (shares Y / Y_truth / POD arrays with the original).
    """
    if (sensor_idx is None) == (n_sensors is None):
        raise ValueError("Provide exactly one of sensor_idx or n_sensors.")

    phi        = dataset["pod_basis"]
    train_mean = dataset["train_mean"]
    scaler     = dataset["scaler"]
    seq_len    = dataset["seq_len"]
    raw_parts  = dataset["raw_parts"]
    n_x        = phi.shape[0]

    if sensor_idx is not None:
        sidx = np.sort(np.asarray(sensor_idx, dtype=np.int32))
        if sidx.ndim != 1 or len(sidx) == 0:
            raise ValueError("sensor_idx must be a non-empty 1-D array of indices.")
        if sidx.min() < 0 or sidx.max() >= n_x:
            raise ValueError(
                f"sensor_idx values must be in [0, {n_x - 1}], "
                f"got min={sidx.min()}, max={sidx.max()}."
            )
    else:
        _, _, pivot_idx = qr(phi @ phi.T, pivoting=True)
        sidx = np.sort(pivot_idx[:n_sensors]).astype(np.int32)

    def _rebuild_x(parts, split_name):
        x_parts = []
        for part_idx, part in enumerate(parts):
            x = part[:, sidx].astype(np.float32)
            if seq_len is not None:
                x = _truncate_and_reshape_split(x, seq_len=seq_len,
                                                split_name=f"{split_name}[{part_idx}]")
            x_parts.append(x)
        return np.concatenate(x_parts, axis=0)

    new_dataset = dict(dataset)   # shallow copy — Y/Y_truth/POD arrays are shared
    new_dataset["sensor_idx"] = sidx
    new_dataset["train"] = dict(dataset["train"], X=_rebuild_x(raw_parts["train"], "train"))
    new_dataset["valid"] = dict(dataset["valid"], X=_rebuild_x(raw_parts["valid"], "valid"))
    new_dataset["test"]  = dict(dataset["test"],  X=_rebuild_x(raw_parts["test"],  "test"))
    return new_dataset


def save_dataset_npz(dataset: Dict[str, Any], output_path: str | Path) -> None:
    """Persist prepared arrays and scaler parameters to an .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scaler: StandardScaler = dataset["scaler"]
    np.savez_compressed(
        output_path,
        sensor_idx=dataset["sensor_idx"],
        pod_basis=dataset["pod_basis"],
        singular_values=dataset["singular_values"],
        train_mean=dataset["train_mean"],
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        X_train=dataset["train"]["X"],
        Y_train=dataset["train"]["Y"],
        Y_train_truth=dataset["train"]["Y_truth"],
        X_valid=dataset["valid"]["X"],
        Y_valid=dataset["valid"]["Y"],
        Y_valid_truth=dataset["valid"]["Y_truth"],
        X_test=dataset["test"]["X"],
        Y_test=dataset["test"]["Y"],
        Y_test_truth=dataset["test"]["Y_truth"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare POD sensing dataset from one or more CSV files.")
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more input CSV file paths.",
    )
    parser.add_argument("--train-ratio", type=float, required=True, help="Train split ratio.")
    parser.add_argument("--valid-ratio", type=float, required=True, help="Validation split ratio.")
    parser.add_argument("--n-sensors", type=int, default=5, help="Number of QDEIM sensors.")
    parser.add_argument("--n-modes", type=int, default=12, help="Number of POD modes.")
    parser.add_argument(
        "--reconstruction-treshold",
        type=float,
        default=None,
        help=(
            "Maximum reconstruction error percentage allowed for train/valid/test. "
            "If set, POD modes are increased until this target is met."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random state for randomized SVD.")
    parser.add_argument(
        "--se-len",
        type=int,
        default=None,
        help="Optional sequence length for train/valid/test reshaping.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal sampling stride (2 means every other timestep).",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Use raw CSV orientation without transpose.",
    )
    parser.add_argument(
        "--output",
        default="processed_data/sensing_dataset.npz",
        help="Output .npz path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = prepare_sensing_data(
        csv_path=args.csv,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        n_sensors=args.n_sensors,
        n_modes=args.n_modes,
        random_state=args.seed,
        transpose=not args.no_transpose,
        seq_len=args.seq_len,
        stride=args.stride,
        reconstruction_treshold=args.reconstruction_treshold,
    )
    save_dataset_npz(dataset, args.output)

    print(f"Saved dataset to: {args.output}")
    print(f"Source files: {len(dataset['source_files'])}")
    print(f"Sensors: {dataset['sensor_idx'].tolist()}")
    print(f"X_train: {dataset['train']['X'].shape}, Y_train: {dataset['train']['Y'].shape}")
    print(f"X_valid: {dataset['valid']['X'].shape}, Y_valid: {dataset['valid']['Y'].shape}")
    print(f"X_test:  {dataset['test']['X'].shape}, Y_test:  {dataset['test']['Y'].shape}")


if __name__ == "__main__":
    main()
