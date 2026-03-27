"""Animation helpers for VIV model comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


# ════════════════════════════════════════════════════════════════
#  Field-comparison animation  (LSSL vs SHRED spatial profiles)
# ════════════════════════════════════════════════════════════════

def make_animation(
    true_seq: np.ndarray,
    pred_seq_lssl: np.ndarray,
    pred_seq_shred: np.ndarray,
    sensor_idx: np.ndarray,
    anim_steps: int = 100,
    title_suffix: str = "",
    display_html: bool = True,
):
    """
    3-panel animation: spatial profile | error trajectory | sensor values.

    Parameters
    ----------
    true_seq       : (T, Nx) ground-truth field
    pred_seq_lssl  : (T, Nx) LSSL prediction
    pred_seq_shred : (T, Nx) SHRED prediction
    sensor_idx     : (S,) spatial sensor indices
    anim_steps     : max frames
    title_suffix   : e.g. '(Test Set)'
    display_html   : if True, try to display in notebook via IPython
    """
    matplotlib.rcParams["animation.embed_limit"] = 100  # MB

    err_seq_t_lssl = np.sqrt(np.mean((pred_seq_lssl - true_seq) ** 2, axis=1))
    err_seq_t_shred = np.sqrt(np.mean((pred_seq_shred - true_seq) ** 2, axis=1))
    sensor_true = true_seq[:, sensor_idx]
    mean_true = sensor_true.mean(axis=1)
    x = np.arange(true_seq.shape[1])
    tt = np.arange(true_seq.shape[0])
    n_frames = min(len(tt), anim_steps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ax0, ax1, ax2 = axes

    v = 1.05 * max(
        np.abs(true_seq).max(),
        np.abs(pred_seq_lssl).max(),
        np.abs(pred_seq_shred).max(),
    )
    (line_true,) = ax0.plot(x, true_seq[0], "k-", lw=2, label="Ground Truth")
    (line_pred_lssl,) = ax0.plot(
        x, pred_seq_lssl[0], color="tab:blue", lw=1.6, label="LSSL prediction"
    )
    (line_pred_shred,) = ax0.plot(
        x, pred_seq_shred[0], color="tab:green", lw=1.6, label="SHRED prediction"
    )
    pts_true = ax0.scatter(
        sensor_idx, true_seq[0, sensor_idx], color="k", s=28, zorder=4, label="True sensors"
    )
    pts_pred_lssl = ax0.scatter(
        sensor_idx, pred_seq_lssl[0, sensor_idx],
        color="tab:blue", s=24, marker="x", zorder=5, label="LSSL Pred sensors",
    )
    pts_pred_shred = ax0.scatter(
        sensor_idx, pred_seq_shred[0, sensor_idx],
        color="tab:green", s=24, marker="x", zorder=5, label="SHRED Pred sensors",
    )
    ax0.set_xlim(x.min(), x.max())
    ax0.set_ylim(-v, v)
    ax0.set_title(f"Spatial Profile {title_suffix}")
    ax0.legend(loc="upper right")

    ax1.plot(tt[:n_frames], err_seq_t_lssl[:n_frames], color="tab:blue", alpha=0.25)
    ax1.plot(tt[:n_frames], err_seq_t_shred[:n_frames], color="tab:green", alpha=0.25)
    (line_err_hist_lssl,) = ax1.plot([], [], color="tab:blue", lw=2, label="RMSE(t) LSSL")
    (line_err_hist_shred,) = ax1.plot([], [], color="tab:green", lw=2, label="RMSE(t) SHRED")
    (pt_err_lssl,) = ax1.plot([tt[0]], [err_seq_t_lssl[0]], "o", color="tab:red", ms=5)
    (pt_err_shred,) = ax1.plot([tt[0]], [err_seq_t_shred[0]], "o", color="tab:orange", ms=5)
    ax1.set_xlim(tt[:n_frames].min(), tt[:n_frames].max())
    ax1.set_ylim(
        0.0,
        1.05 * max(err_seq_t_lssl[:n_frames].max(), err_seq_t_shred[:n_frames].max(), 1e-8),
    )
    ax1.set_title("Error Trajectory")
    ax1.legend(loc="upper right")

    for j in range(len(sensor_idx)):
        ax2.plot(tt[:n_frames], sensor_true[:n_frames, j], color="k", alpha=0.18, lw=1)
    (line_s_true,) = ax2.plot(tt[:n_frames], mean_true[:n_frames], "k-", lw=2, label="Mean sensor true")
    line_t_sensor = ax2.axvline(tt[0], color="tab:red", ls="--", lw=1.5)
    ys = sensor_true[:n_frames].reshape(-1)
    pad = 0.05 * (ys.max() - ys.min() + 1e-8)
    ax2.set_xlim(tt[:n_frames].min(), tt[:n_frames].max())
    ax2.set_ylim(ys.min() - pad, ys.max() + pad)
    ax2.set_title("Sensor Values Over Time")
    ax2.set_xlabel("time in sequence")
    ax2.legend(loc="upper right")

    time_text = fig.text(0.5, 0.01, "", ha="center")

    def init():
        line_true.set_ydata(true_seq[0])
        line_pred_lssl.set_ydata(pred_seq_lssl[0])
        line_pred_shred.set_ydata(pred_seq_shred[0])
        pts_true.set_offsets(np.c_[sensor_idx, true_seq[0, sensor_idx]])
        pts_pred_lssl.set_offsets(np.c_[sensor_idx, pred_seq_lssl[0, sensor_idx]])
        pts_pred_shred.set_offsets(np.c_[sensor_idx, pred_seq_shred[0, sensor_idx]])
        line_err_hist_lssl.set_data([tt[0]], [err_seq_t_lssl[0]])
        line_err_hist_shred.set_data([tt[0]], [err_seq_t_shred[0]])
        pt_err_lssl.set_data([tt[0]], [err_seq_t_lssl[0]])
        pt_err_shred.set_data([tt[0]], [err_seq_t_shred[0]])
        line_t_sensor.set_xdata([tt[0], tt[0]])
        time_text.set_text("t = 0")
        return [
            line_true, line_pred_lssl, line_pred_shred,
            pts_true, pts_pred_lssl, pts_pred_shred,
            line_err_hist_lssl, line_err_hist_shred,
            pt_err_lssl, pt_err_shred,
            line_t_sensor, line_s_true, time_text,
        ]

    def update(frame):
        line_true.set_ydata(true_seq[frame])
        line_pred_lssl.set_ydata(pred_seq_lssl[frame])
        line_pred_shred.set_ydata(pred_seq_shred[frame])
        pts_true.set_offsets(np.c_[sensor_idx, true_seq[frame, sensor_idx]])
        pts_pred_lssl.set_offsets(np.c_[sensor_idx, pred_seq_lssl[frame, sensor_idx]])
        pts_pred_shred.set_offsets(np.c_[sensor_idx, pred_seq_shred[frame, sensor_idx]])
        line_err_hist_lssl.set_data(tt[: frame + 1], err_seq_t_lssl[: frame + 1])
        line_err_hist_shred.set_data(tt[: frame + 1], err_seq_t_shred[: frame + 1])
        pt_err_lssl.set_data([tt[frame]], [err_seq_t_lssl[frame]])
        pt_err_shred.set_data([tt[frame]], [err_seq_t_shred[frame]])
        line_t_sensor.set_xdata([tt[frame], tt[frame]])
        time_text.set_text(f"t = {frame}/{n_frames - 1}")
        return [
            line_true, line_pred_lssl, line_pred_shred,
            pts_true, pts_pred_lssl, pts_pred_shred,
            line_err_hist_lssl, line_err_hist_shred,
            pt_err_lssl, pt_err_shred,
            line_t_sensor, line_s_true, time_text,
        ]

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=80, blit=True, repeat=True
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if display_html:
        try:
            from IPython.display import HTML, display, Image

            display(HTML(ani.to_jshtml(fps=30, default_mode="once")))
        except Exception as e:
            print("jshtml failed, falling back to GIF:", e)
            gif_path = Path(f"figures/viv_anim{title_suffix.replace(' ', '_')}.gif")
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(gif_path, writer=animation.PillowWriter(fps=30))
            display(Image(filename=str(gif_path)))
    plt.close(fig)
    return ani


# ════════════════════════════════════════════════════════════════
#  POD-mode coefficient animation  (LSSL vs SHRED per-mode)
# ════════════════════════════════════════════════════════════════

def make_modes_animation(
    y_true_seq: np.ndarray,
    y_pred_seq_lssl: np.ndarray,
    y_pred_seq_shred: np.ndarray,
    anim_steps: int = 100,
    split_name: str = "train",
    display_html: bool = True,
):
    """
    3-panel mode-coefficient animation: coefficients | total RMSE | per-mode error.

    Parameters
    ----------
    y_true_seq       : (T, n_modes)
    y_pred_seq_lssl  : (T, n_modes)
    y_pred_seq_shred : (T, n_modes)
    anim_steps       : max frames
    split_name       : label for title ('train', 'valid', 'test')
    display_html     : if True, try to display in notebook via IPython
    """
    n_timesteps, n_modes = y_true_seq.shape
    n_timesteps = min(n_timesteps, anim_steps)

    error_seq_t_lssl = np.sqrt(np.mean((y_pred_seq_lssl - y_true_seq) ** 2, axis=1))
    error_seq_t_shred = np.sqrt(np.mean((y_pred_seq_shred - y_true_seq) ** 2, axis=1))

    mode_idx = np.arange(n_modes)
    tt = np.arange(n_timesteps)

    print(f"Animating {n_timesteps} frames for {split_name} set")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ax0, ax1, ax2 = axes

    v = 1.05 * max(
        np.abs(y_true_seq).max(),
        np.abs(y_pred_seq_lssl).max(),
        np.abs(y_pred_seq_shred).max(),
    )
    (line_true,) = ax0.plot(mode_idx, y_true_seq[0], "ko-", lw=2, markersize=6, label="True")
    (line_pred_lssl,) = ax0.plot(
        mode_idx, y_pred_seq_lssl[0], "b^-", lw=2, markersize=5, label="Predicted LSSL"
    )
    (line_pred_shred,) = ax0.plot(
        mode_idx, y_pred_seq_shred[0], "gs-", lw=2, markersize=5, label="Predicted SHRED"
    )
    ax0.set_xlim(mode_idx.min() - 0.5, mode_idx.max() + 0.5)
    ax0.set_ylim(-v, v)
    ax0.set_xlabel("Mode Index")
    ax0.set_ylabel("Coefficient Value")
    ax0.set_title(f"Mode Coefficients {split_name.upper()}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    ax1.plot(tt, error_seq_t_lssl[:n_timesteps], color="tab:blue", alpha=0.25)
    ax1.plot(tt, error_seq_t_shred[:n_timesteps], color="tab:green", alpha=0.25)
    (line_err_hist_lssl,) = ax1.plot([], [], color="tab:blue", lw=2, label="RMSE LSSL")
    (line_err_hist_shred,) = ax1.plot([], [], color="tab:green", lw=2, label="RMSE SHRED")
    (pt_err_lssl,) = ax1.plot([tt[0]], [error_seq_t_lssl[0]], "o", color="tab:red", ms=6)
    (pt_err_shred,) = ax1.plot([tt[0]], [error_seq_t_shred[0]], "o", color="tab:orange", ms=6)
    ax1.set_ylim(
        0.0,
        1.05 * max(error_seq_t_lssl.max(), error_seq_t_shred.max(), 1e-8),
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Total Error Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    for m in range(n_modes):
        ax2.plot(tt, np.abs(y_pred_seq_lssl[:n_timesteps, m] - y_true_seq[:n_timesteps, m]), alpha=0.4, lw=0.8)
        ax2.plot(tt, np.abs(y_pred_seq_shred[:n_timesteps, m] - y_true_seq[:n_timesteps, m]), alpha=0.4, lw=0.8)

    (pt_mode_error,) = ax2.plot([], [], "o", color="tab:red", ms=8, label="Current errors")
    ax2.set_xlim(tt.min(), tt.max())
    ax2.set_ylim(
        0,
        max(
            np.abs(y_pred_seq_lssl[:n_timesteps] - y_true_seq[:n_timesteps]).max(),
            np.abs(y_pred_seq_shred[:n_timesteps] - y_true_seq[:n_timesteps]).max(),
        )
        * 1.1,
    )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Individual Mode Errors")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=11)

    def init():
        line_true.set_ydata(y_true_seq[0])
        line_pred_lssl.set_ydata(y_pred_seq_lssl[0])
        line_pred_shred.set_ydata(y_pred_seq_shred[0])
        line_err_hist_lssl.set_data([tt[0]], [error_seq_t_lssl[0]])
        line_err_hist_shred.set_data([tt[0]], [error_seq_t_shred[0]])
        pt_err_lssl.set_data([tt[0]], [error_seq_t_lssl[0]])
        pt_err_shred.set_data([tt[0]], [error_seq_t_shred[0]])
        pt_mode_error.set_data([], [])
        time_text.set_text("t = 0")
        return [
            line_true, line_pred_lssl, line_pred_shred,
            line_err_hist_lssl, line_err_hist_shred,
            pt_err_lssl, pt_err_shred, pt_mode_error, time_text,
        ]

    def update(frame):
        line_true.set_ydata(y_true_seq[frame])
        line_pred_lssl.set_ydata(y_pred_seq_lssl[frame])
        line_pred_shred.set_ydata(y_pred_seq_shred[frame])
        line_err_hist_lssl.set_data(tt[: frame + 1], error_seq_t_lssl[: frame + 1])
        line_err_hist_shred.set_data(tt[: frame + 1], error_seq_t_shred[: frame + 1])
        pt_err_lssl.set_data([tt[frame]], [error_seq_t_lssl[frame]])
        pt_err_shred.set_data([tt[frame]], [error_seq_t_shred[frame]])
        current_errors_lssl = np.abs(y_pred_seq_lssl[frame] - y_true_seq[frame])
        error_times = np.full(n_modes, frame)
        pt_mode_error.set_data(error_times, current_errors_lssl)
        time_text.set_text(f"t = {frame}/{n_timesteps - 1}")
        return [
            line_true, line_pred_lssl, line_pred_shred,
            line_err_hist_lssl, line_err_hist_shred,
            pt_err_lssl, pt_err_shred, pt_mode_error, time_text,
        ]

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_timesteps, interval=80, blit=True, repeat=True
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if display_html:
        try:
            from IPython.display import HTML, display, Image

            display(HTML(ani.to_jshtml(fps=30, default_mode="once")))
        except Exception as e:
            print("jshtml failed, falling back to GIF:", e)
            gif_path = Path(f"figures/viv_modes_anim_{split_name}.gif")
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(gif_path, writer=animation.PillowWriter(fps=30))
            display(Image(filename=str(gif_path)))
    plt.close(fig)
    return ani
