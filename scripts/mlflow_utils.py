"""MLflow logging utilities for model tracking and comparison."""

import mlflow
import mlflow.pytorch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import jax.numpy as jnp
import equinox as eqx


def start_mlflow_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Dict[str, str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """
    Start a new MLflow run.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name for this specific run.  If *None* (default), a name
            is auto-generated as ``YYYY-MM-DD_vN`` where *N* is the next
            version number for today (e.g. ``2026-02-10_v1``,
            ``2026-02-10_v2``, …).
        tags: Optional dictionary of tags to add to the run
        project_root: Project root directory. If provided, the tracking URI
            is set to ``file:///<project_root>/mlruns`` so that all runs are
            stored in the same filesystem-based backend regardless of CWD.
        
    Returns:
        Tuple of (run_name, run_id).
    """
    # Pin tracking URI to project-root mlruns/ folder so runs never end up
    # in a stray SQLite DB when the notebook CWD differs from the project root.
    if project_root is not None:
        tracking_uri = Path(project_root).resolve() / "mlruns"
        mlflow.set_tracking_uri(f"file:///{tracking_uri}")

    # Set experiment first so we can query existing runs
    mlflow.set_experiment(experiment_name)

    # Auto-generate a date + version run name
    if run_name is None:
        today = datetime.now().strftime("%Y-%m-%d")
        prefix = f"{today}_v"

        # Find the highest version for today in this experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        max_version = 0
        if experiment is not None:
            from mlflow.entities import ViewType
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName LIKE '{prefix}%'",
                run_view_type=ViewType.ALL,
            )
            for name in runs.get("tags.mlflow.runName", []):
                try:
                    v = int(name.split("_v")[-1])
                    max_version = max(max_version, v)
                except (ValueError, IndexError):
                    pass

        run_name = f"{prefix}{max_version + 1}"
    
    # Start run
    run = mlflow.start_run(run_name=run_name)
    
    # Log tags if provided
    if tags:
        mlflow.set_tags(tags)

    # Create a human-readable symlink so the mlruns folder is browsable
    if project_root is not None:
        _create_run_symlink(
            project_root=Path(project_root),
            experiment_id=run.info.experiment_id,
            run_id=run.info.run_id,
            run_name=run_name,
        )

    print(f"MLflow run started: {run_name}  (id={run.info.run_id[:8]}…)")
    return run_name, run.info.run_id


def _create_run_symlink(
    project_root: Path,
    experiment_id: str,
    run_id: str,
    run_name: str,
) -> None:
    """Create a symlink ``mlruns_named/<run_name> -> mlruns/<exp>/<run_id>``."""
    named_dir = project_root / "mlruns_named"
    named_dir.mkdir(exist_ok=True)

    target = project_root / "mlruns" / experiment_id / run_id
    link = named_dir / run_name

    # Remove stale link if it exists (e.g. re-running the same notebook)
    if link.is_symlink() or link.exists():
        link.unlink()

    try:
        link.symlink_to(target)
    except OSError:
        pass  # silently skip on systems that don't support symlinks


def log_dataset_params(
    csv_dir: Path,
    pair_index: int,
    cf_path: str,
    il_path: str,
    lag: int,
    nsensors: int,
    modes_cf: int,
    modes_il: int,
    seed: int,
) -> None:
    """Log dataset configuration parameters."""
    params = {
        "csv_dir": str(csv_dir),
        "pair_index": pair_index,
        "cf_data_file": Path(cf_path).name,
        "il_data_file": Path(il_path).name,
        "lag": lag,
        "n_sensors": nsensors,
        "modes_cf": modes_cf,
        "modes_il": modes_il,
        "seed": seed,
    }
    mlflow.log_params(params)


def log_cde_model_params(
    hidden_size: int,
    width_size: int,
    depth: int,
    output_size: int,
    decoder_sizes: list,
) -> None:
    """Log Neural CDE model architecture parameters."""
    params = {
        "CDE_model_type": "NeuralCDE",
        "CDE_hidden_size": hidden_size,
        "CDE_width_size": width_size,
        "CDE_depth": depth,
        "CDE_output_size": output_size,
        "CDE_decoder_sizes": ",".join(map(str, decoder_sizes)),
        "CDE_activation_cde": "gelu",
        "CDE_activation_decoder": "gelu",
    }
    mlflow.log_params(params)


def log_shred_model_params(
    input_size: int,
    hidden_size: int,
    hidden_layers: int,
    output_size: int,
    decoder_sizes: list,
) -> None:
    """Log SHRED model architecture parameters."""
    params = {
        "SHRED_model_type": "SHRED",
        "SHRED_input_size": input_size,
        "SHRED_hidden_size": hidden_size,
        "SHRED_hidden_layers": hidden_layers,
        "SHRED_output_size": output_size,
        "SHRED_decoder_sizes": ",".join(map(str, decoder_sizes)),
        "SHRED_activation": "gelu",
    }
    mlflow.log_params(params)


def log_training_params(
    epochs: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
    model_type: str = "CDE",
) -> None:
    """Log training hyperparameters."""
    params = {
        f"{model_type}_epochs": epochs,
        f"{model_type}_learning_rate": learning_rate,
        f"{model_type}_batch_size": batch_size,
        f"{model_type}_early_stopping_patience": patience,
    }
    mlflow.log_params(params)


def log_training_metrics(
    train_losses: list,
    valid_losses: list,
    model_type: str = "CDE",
) -> None:
    """
    Log training metrics from loss history.
    
    Args:
        train_losses: List of training losses at each epoch
        valid_losses: List of validation losses at each epoch
        model_type: Type of model ("CDE" or "SHRED")
    """
    # Log final losses
    final_train_loss = float(train_losses[-1]) if isinstance(train_losses[-1], jnp.ndarray) else train_losses[-1]
    final_valid_loss = float(valid_losses[-1]) if isinstance(valid_losses[-1], jnp.ndarray) else valid_losses[-1]
    final_best_valid = float(min(valid_losses))
    
    metrics = {
        f"{model_type}_final_train_loss": final_train_loss,
        f"{model_type}_final_valid_loss": final_valid_loss,
        f"{model_type}_best_valid_loss": final_best_valid,
        f"{model_type}_n_epochs_trained": len(train_losses),
    }
    mlflow.log_metrics(metrics)


def log_evaluation_metrics(
    preds_cf_train: jnp.ndarray,
    preds_cf_valid: jnp.ndarray,
    preds_cf_test: jnp.ndarray,
    preds_cf_full: jnp.ndarray,
    preds_il_train: jnp.ndarray,
    preds_il_valid: jnp.ndarray,
    preds_il_test: jnp.ndarray,
    preds_il_full: jnp.ndarray,
    full_cf_data: jnp.ndarray,
    full_il_data: jnp.ndarray,
    model_type: str = "CDE",
) -> Dict[str, float]:
    """
    Calculate and log evaluation metrics (RMSE, etc.) for full dataset.
    
    Args:
        preds_cf_train, preds_cf_valid, preds_cf_test: Cross-flow predictions
        preds_cf_full: Full cross-flow predictions
        preds_il_train, preds_il_valid, preds_il_test: In-line predictions
        preds_il_full: Full in-line predictions
        full_cf_data: Full cross-flow ground truth
        full_il_data: Full in-line ground truth
        model_type: Type of model ("CDE" or "SHRED")
        
    Returns:
        Dictionary of computed metrics
    """
    def rmsre(pred, true):
        """Root Mean Square Relative Error."""
        return jnp.mean(jnp.sqrt(jnp.sum((pred - true) ** 2, axis=-1)) / 
                        jnp.sqrt(jnp.sum(true ** 2, axis=-1)))
    
    def rmse(pred, true):
        """Root Mean Square Error."""
        return jnp.sqrt(jnp.mean((pred - true) ** 2))
    
    def mae(pred, true):
        """Mean Absolute Error."""
        return jnp.mean(jnp.abs(pred - true))
    
    # Calculate metrics on full sequence
    cf_rmsre = float(rmsre(preds_cf_full, full_cf_data))
    il_rmsre = float(rmsre(preds_il_full, full_il_data))
    cf_rmse = float(rmse(preds_cf_full, full_cf_data))
    il_rmse = float(rmse(preds_il_full, full_il_data))
    cf_mae = float(mae(preds_cf_full, full_cf_data))
    il_mae = float(mae(preds_il_full, full_il_data))
    
    metrics = {
        f"{model_type}_cf_rmsre_full": cf_rmsre,
        f"{model_type}_il_rmsre_full": il_rmsre,
        f"{model_type}_cf_rmse_full": cf_rmse,
        f"{model_type}_il_rmse_full": il_rmse,
        f"{model_type}_cf_mae_full": cf_mae,
        f"{model_type}_il_mae_full": il_mae,
    }
    
    mlflow.log_metrics(metrics)
    return metrics


def log_split_metrics(
    preds_cf_train: jnp.ndarray,
    preds_cf_valid: jnp.ndarray,
    preds_cf_test: jnp.ndarray,
    preds_il_train: jnp.ndarray,
    preds_il_valid: jnp.ndarray,
    preds_il_test: jnp.ndarray,
    true_cf_train: jnp.ndarray,
    true_cf_valid: jnp.ndarray,
    true_cf_test: jnp.ndarray,
    true_il_train: jnp.ndarray,
    true_il_valid: jnp.ndarray,
    true_il_test: jnp.ndarray,
    model_type: str = "CDE",
) -> Dict[str, float]:
    """
    Calculate and log evaluation metrics per data split.
    
    Args:
        preds_*: Model predictions for each split and field
        true_*: Ground truth for each split and field
        model_type: Type of model ("CDE" or "SHRED")
        
    Returns:
        Dictionary of computed metrics
    """
    def rmse(pred, true):
        """Root Mean Square Error."""
        return jnp.sqrt(jnp.mean((pred - true) ** 2))
    
    metrics = {
        f"{model_type}_cf_rmse_train": float(rmse(preds_cf_train, true_cf_train)),
        f"{model_type}_cf_rmse_valid": float(rmse(preds_cf_valid, true_cf_valid)),
        f"{model_type}_cf_rmse_test": float(rmse(preds_cf_test, true_cf_test)),
        f"{model_type}_il_rmse_train": float(rmse(preds_il_train, true_il_train)),
        f"{model_type}_il_rmse_valid": float(rmse(preds_il_valid, true_il_valid)),
        f"{model_type}_il_rmse_test": float(rmse(preds_il_test, true_il_test)),
    }
    
    mlflow.log_metrics(metrics)
    return metrics


def count_model_params(model) -> int:
    """Count total trainable parameters in a JAX/Equinox model."""
    total = 0
    for leaf in jax.tree.leaves(model):
        if isinstance(leaf, jnp.ndarray):
            total += leaf.size
    return total


def log_model_architecture_info(
    model,
    model_type: str = "CDE",
) -> None:
    """Log model architecture information (parameter counts, etc.)."""
    param_count = count_model_params(model)
    
    metrics = {
        f"{model_type}_total_parameters": param_count,
    }
    
    mlflow.log_metrics(metrics)


def get_loss_history_summary(losses: list) -> Dict[str, float]:
    """Get summary statistics of loss history."""
    losses_array = jnp.array(losses)
    return {
        "min": float(jnp.min(losses_array)),
        "max": float(jnp.max(losses_array)),
        "mean": float(jnp.mean(losses_array)),
        "std": float(jnp.std(losses_array)),
        "final": float(losses_array[-1]),
    }


def log_loss_history_summary(
    train_losses: list,
    valid_losses: list,
    model_type: str = "CDE",
) -> None:
    """Log summary statistics of loss history."""
    train_summary = get_loss_history_summary(train_losses)
    valid_summary = get_loss_history_summary(valid_losses)
    
    metrics = {
        f"{model_type}_train_loss_min": train_summary["min"],
        f"{model_type}_train_loss_max": train_summary["max"],
        f"{model_type}_train_loss_mean": train_summary["mean"],
        f"{model_type}_train_loss_std": train_summary["std"],
        f"{model_type}_valid_loss_min": valid_summary["min"],
        f"{model_type}_valid_loss_max": valid_summary["max"],
        f"{model_type}_valid_loss_mean": valid_summary["mean"],
        f"{model_type}_valid_loss_std": valid_summary["std"],
    }
    
    mlflow.log_metrics(metrics)


def log_comparison_metrics(
    cde_metrics: Dict[str, float],
    shred_metrics: Dict[str, float],
) -> None:
    """Log comparison metrics between CDE and SHRED models."""
    # Calculate relative improvements
    cf_improvement = ((shred_metrics.get("SHRED_cf_rmsre_full", 0) - 
                       cde_metrics.get("CDE_cf_rmsre_full", 0)) / 
                      shred_metrics.get("SHRED_cf_rmsre_full", 1)) * 100 if "SHRED_cf_rmsre_full" in shred_metrics else 0
    
    il_improvement = ((shred_metrics.get("SHRED_il_rmsre_full", 0) - 
                       cde_metrics.get("CDE_il_rmsre_full", 0)) / 
                      shred_metrics.get("SHRED_il_rmsre_full", 1)) * 100 if "SHRED_il_rmsre_full" in shred_metrics else 0
    
    comparison_metrics = {
        "CDE_vs_SHRED_cf_improvement_pct": cf_improvement,
        "CDE_vs_SHRED_il_improvement_pct": il_improvement,
    }
    
    mlflow.log_metrics(comparison_metrics)


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()


import jax
