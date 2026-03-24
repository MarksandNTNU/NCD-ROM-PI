"""Utilities for analyzing and comparing MLflow runs."""

import mlflow
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json


def get_all_runs(experiment_name: str) -> pd.DataFrame:
    """
    Get all runs from an experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        DataFrame with run information
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


def compare_runs(
    experiment_name: str,
    metrics_to_compare: List[str] = None,
) -> pd.DataFrame:
    """
    Compare metrics across runs in an experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metrics_to_compare: List of metric names to compare (if None, all metrics are shown)
        
    Returns:
        DataFrame with comparison results
    """
    runs = get_all_runs(experiment_name)
    
    if runs is None or runs.empty:
        return None
    
    # Extract metrics and hyperparameters
    comparison_data = []
    for _, run in runs.iterrows():
        run_info = {
            "run_id": run["run_id"],
            "run_date": run["start_time"],
            "status": run["status"],
        }
        
        # Add metrics
        for col in runs.columns:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                if metrics_to_compare is None or metric_name in metrics_to_compare:
                    run_info[metric_name] = run[col]
        
        # Add params
        for col in runs.columns:
            if col.startswith("params."):
                param_name = col.replace("params.", "")
                run_info[param_name] = run[col]
        
        comparison_data.append(run_info)
    
    return pd.DataFrame(comparison_data)


def print_run_comparison(
    experiment_name: str,
    model_types: List[str] = ["CDE", "SHRED"],
) -> None:
    """
    Print a formatted comparison of runs.
    
    Args:
        experiment_name: Name of the MLflow experiment
        model_types: Model types to compare
    """
    runs = get_all_runs(experiment_name)
    
    if runs is None or runs.empty:
        print(f"No runs found for experiment '{experiment_name}'")
        return
    
    print(f"\n{'='*100}")
    print(f"MLflow Experiment: {experiment_name}")
    print(f"{'='*100}\n")
    
    for _, run in runs.iterrows():
        print(f"Run ID: {run['run_id']}")
        print(f"Status: {run['status']}")
        print(f"Date: {run['start_time']}\n")
        
        # Print metrics
        print("Metrics:")
        for col in runs.columns:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                value = run[col]
                if value is not None:
                    print(f"  {metric_name}: {value:.6f}")
        
        print("\nParameters:")
        for col in runs.columns:
            if col.startswith("params."):
                param_name = col.replace("params.", "")
                value = run[col]
                if value is not None:
                    print(f"  {param_name}: {value}")
        
        print(f"\n{'-'*100}\n")


def export_comparison_csv(
    experiment_name: str,
    output_path: str = "mlflow_comparison.csv",
) -> None:
    """
    Export run comparison to CSV file.
    
    Args:
        experiment_name: Name of the MLflow experiment
        output_path: Path to save the CSV file
    """
    comparison_df = compare_runs(experiment_name)
    
    if comparison_df is not None:
        comparison_df.to_csv(output_path, index=False)
        print(f"Comparison saved to {output_path}")
    else:
        print(f"No data to export for experiment '{experiment_name}'")


def get_best_model_run(
    experiment_name: str,
    metric_name: str,
    model_type: str = "CDE",
    mode: str = "min",  # 'min' for error metrics, 'max' for performance metrics
) -> Dict:
    """
    Get the best run based on a specific metric.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric_name: Name of the metric to optimize
        model_type: Type of model to filter by
        mode: 'min' or 'max' for metric optimization
        
    Returns:
        Dictionary with best run information
    """
    runs = get_all_runs(experiment_name)
    
    if runs is None or runs.empty:
        return None
    
    # Filter for relevant metric column
    metric_col = f"metrics.{model_type}_{metric_name}"
    
    if metric_col not in runs.columns:
        print(f"Metric '{metric_name}' not found for model type '{model_type}'")
        return None
    
    # Remove NaN values
    valid_runs = runs[runs[metric_col].notna()]
    
    if valid_runs.empty:
        return None
    
    # Find best run
    if mode == "min":
        best_run = valid_runs.loc[valid_runs[metric_col].idxmin()]
    else:
        best_run = valid_runs.loc[valid_runs[metric_col].idxmax()]
    
    return {
        "run_id": best_run["run_id"],
        "metric_value": best_run[metric_col],
        "params": {col.replace("params.", ""): best_run[col] 
                   for col in runs.columns if col.startswith("params.")},
        "metrics": {col.replace("metrics.", ""): best_run[col] 
                    for col in runs.columns if col.startswith("metrics.")}
    }


def summarize_experiment(experiment_name: str) -> None:
    """
    Print a summary of an experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    runs = get_all_runs(experiment_name)
    
    if runs is None or runs.empty:
        print(f"No runs found for experiment '{experiment_name}'")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment Summary: {experiment_name}")
    print(f"{'='*80}")
    print(f"Total Runs: {len(runs)}")
    print(f"Completed Runs: {len(runs[runs['status'] == 'FINISHED'])}")
    print(f"Failed Runs: {len(runs[runs['status'] == 'FAILED'])}")
    
    # Show best models per metric
    print(f"\n{'Best Models by Metric':^80}")
    print(f"{'-'*80}")
    
    for model_type in ["CDE", "SHRED"]:
        print(f"\n{model_type}:")
        for metric in ["cf_rmsre_full", "il_rmsre_full"]:
            best_run = get_best_model_run(experiment_name, metric, model_type, mode="min")
            if best_run:
                print(f"  Best {metric}: {best_run['metric_value']:.6f} "
                      f"(Run: {best_run['run_id'][:8]}...)")
