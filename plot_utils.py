"""Plotting and animation utilities for VIV analysis"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Optional


def plot_training_curves(train_losses: np.ndarray, valid_losses: np.ndarray,
                        title: str = "Training Curves") -> tuple:
    """Plot training and validation loss curves
    
    Args:
        train_losses: Training loss history
        valid_losses: Validation loss history
        title: Plot title
        
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(valid_losses, label='Valid', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_spatial_heatmap(data: np.ndarray, title: str = "Spatial-Temporal Data",
                        save_path: Optional[str] = None) -> tuple:
    """Plot spatial-temporal heatmap
    
    Args:
        data: Data array (time x spatial)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower', 
                   interpolation='bilinear')
    ax.set_xlabel('Spatial Point')
    ax.set_ylabel('Time Step')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.tight_layout()
    return fig, ax


def plot_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                              spatial_indices: list, title: str = "Prediction Comparison",
                              save_path: Optional[str] = None) -> tuple:
    """Plot comparison of predictions vs ground truth at specific spatial points
    
    Args:
        y_true: Ground truth data (time x spatial)
        y_pred: Predictions (time x spatial)
        spatial_indices: List of spatial indices to plot
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        fig, axes
    """
    n_plots = len(spatial_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, idx in enumerate(spatial_indices):
        ax = axes[i]
        ax.plot(y_true[:, idx], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(y_pred[:, idx], 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Spatial Point {idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.tight_layout()
    return fig, axes


def animate_spatial_temporal(data: np.ndarray, fps: int = 10,
                            save_path: Optional[str] = None) -> animation.FuncAnimation:
    """Create spatial-temporal animation
    
    Args:
        data: Data array (time x spatial)
        fps: Frames per second
        save_path: Path to save animation
        
    Returns:
        Animation object
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    vmin, vmax = data.min(), data.max()
    im = ax.imshow(data[0:1], aspect='auto', cmap='viridis', 
                   vmin=vmin, vmax=vmax, origin='lower')
    ax.set_ylabel('Spatial Point')
    ax.set_xlabel('Value')
    ax.set_title('Spatial-Temporal Evolution - Frame 0')
    plt.colorbar(im, ax=ax)
    
    def update(frame):
        im.set_data(data[frame:frame+1])
        ax.set_title(f'Spatial-Temporal Evolution - Frame {frame}/{data.shape[0]}')
        return im,
    
    anim = animation.FuncAnimation(fig, update, frames=data.shape[0],
                                  interval=1000//fps, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"✅ Animation saved: {save_path}")
    
    return anim


def animate_comparison(y_true: np.ndarray, y_pred: np.ndarray, fps: int = 10,
                      save_path: Optional[str] = None) -> animation.FuncAnimation:
    """Create side-by-side animation of predictions vs ground truth
    
    Args:
        y_true: Ground truth data (time x spatial)
        y_pred: Predictions (time x spatial)
        fps: Frames per second
        save_path: Path to save animation
        
    Returns:
        Animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    
    im1 = ax1.imshow(y_true[0:1], aspect='auto', cmap='RdBu_r',
                     vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title('Ground Truth - Frame 0')
    ax1.set_ylabel('Spatial Point')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(y_pred[0:1], aspect='auto', cmap='RdBu_r',
                     vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title('Prediction - Frame 0')
    ax2.set_ylabel('Spatial Point')
    plt.colorbar(im2, ax=ax2)
    
    def update(frame):
        im1.set_data(y_true[frame:frame+1])
        im2.set_data(y_pred[frame:frame+1])
        ax1.set_title(f'Ground Truth - Frame {frame}')
        ax2.set_title(f'Prediction - Frame {frame}')
        return im1, im2
    
    anim = animation.FuncAnimation(fig, update,
                                  frames=min(y_true.shape[0], y_pred.shape[0]),
                                  interval=1000//fps, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"✅ Animation saved: {save_path}")
    
    return anim


def plot_residuals(residuals: np.ndarray, title: str = "Residual Errors",
                  save_path: Optional[str] = None) -> tuple:
    """Plot residual/error statistics
    
    Args:
        residuals: Error array (time x spatial)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spatial error distribution
    spatial_error = np.mean(np.abs(residuals), axis=0)
    axes[0].plot(spatial_error, linewidth=2)
    axes[0].set_xlabel('Spatial Point')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('Error Distribution Across Space')
    axes[0].grid(True, alpha=0.3)
    
    # Temporal error distribution
    temporal_error = np.mean(np.abs(residuals), axis=1)
    axes[1].plot(temporal_error, linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Error Distribution Over Time')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.tight_layout()
    return fig, axes
