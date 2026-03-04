"""Animation helpers for VIV model comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import create_comparison_animation


def save_comparison_gif(
    x: np.ndarray,
    sensor_data: np.ndarray,
    shred_prediction: np.ndarray,
    cde_prediction: np.ndarray,
    ground_truth: np.ndarray,
    sensor_locations: np.ndarray,
    output_path: Path | str,
    fps: int = 20,
) -> Path:
    """Save a comparison GIF of SHRED vs CDE predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim = create_comparison_animation(
        x,
        sensor_data,
        shred_prediction,
        cde_prediction,
        ground_truth,
        sensor_locations,
        nsensors=len(sensor_locations),
    )
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close("all")
    return output_path


def smoke_test_animation(output_path: Optional[Path | str] = None) -> object:
    """Create a tiny animation to verify plotting works."""
    rng = np.random.default_rng(0)
    nt = 6
    lag = 4
    nx = 8
    nsensors = 3

    x = np.linspace(0, 1, nx)
    sensor_data = rng.standard_normal((nt, lag, nsensors))
    shred_prediction = rng.standard_normal((nt, nx))
    cde_prediction = rng.standard_normal((nt, nx))
    ground_truth = rng.standard_normal((nt, nx))
    sensor_locations = np.array([0, 3, 7])

    anim = create_comparison_animation(
        x,
        sensor_data,
        shred_prediction,
        cde_prediction,
        ground_truth,
        sensor_locations,
        nsensors=nsensors,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer="pillow", fps=10)

    plt.close("all")
    return anim
