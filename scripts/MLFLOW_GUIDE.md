"""
MLflow Integration Guide for VIV Model Training
================================================

This guide explains how to use the MLflow logging utilities integrated into your workflow.

## Files Created

1. **scripts/mlflow_utils.py**
   - Core MLflow logging functions
   - Called automatically in the notebook
   - Logs hyperparameters, model architecture, training metrics, and evaluation metrics

2. **scripts/mlflow_analysis.py**
   - Tools for analyzing and comparing runs
   - Export results to CSV
   - Find best models by metric

3. **scripts/model_tracking.py**
   - Historical tracking of model performance
   - Compare models across runs
   - Persistent storage in JSON format


## How MLflow Logging Works in Your Notebook

The notebook now automatically logs:

### 1. **Experiment Setup**
   - Experiment name: "VIV_NeuralCDE_vs_SHRED"
   - Tags: dataset, phase

### 2. **Dataset Parameters**
   - CSV directory, pair index
   - Data files (CF and IL)
   - Preprocessing parameters: lag, n_sensors, modes, seed

### 3. **Model Architecture**
   - For CDE: hidden_size, width_size, depth, decoder_sizes
   - For SHRED: input_size, hidden_layers, output_size
   - Total parameter counts

### 4. **Training Hyperparameters**
   - Epochs, learning rate, batch size, early stopping patience
   - Logged separately for CDE and SHRED

### 5. **Training Metrics**
   - Final train/validation loss
   - Best validation loss
   - Loss statistics (min, max, mean, std)

### 6. **Evaluation Metrics**
   - RMSE, RMSRE, MAE on full sequences
   - Split-wise metrics (train/valid/test)
   - Field-specific metrics (CF and IL)
   - Comparison metrics (CDE vs SHRED improvement)


## Using MLflow UI

View results locally:
```bash
cd /path/to/NCD-ROM-new
mlflow ui
```
Then open: http://localhost:5000

This shows:
- All experiment runs
- Parameter values for each run
- Metric trends
- Run comparison tables


## Analysis After Running Notebook

### 1. Print Comparison
```python
from scripts.mlflow_analysis import print_run_comparison

print_run_comparison("VIV_NeuralCDE_vs_SHRED")
```

### 2. Export to CSV
```python
from scripts.mlflow_analysis import export_comparison_csv

export_comparison_csv(
    "VIV_NeuralCDE_vs_SHRED",
    output_path="mlflow_comparison.csv"
)
```

### 3. Find Best Model
```python
from scripts.mlflow_analysis import get_best_model_run

best_cde = get_best_model_run(
    "VIV_NeuralCDE_vs_SHRED",
    "cf_rmsre_full",
    model_type="CDE",
    mode="min"
)

print(f"Best CDE run: {best_cde['run_id']}")
print(f"Best metric value: {best_cde['metric_value']:.6f}")
```

### 4. Experiment Summary
```python
from scripts.mlflow_analysis import summarize_experiment

summarize_experiment("VIV_NeuralCDE_vs_SHRED")
```


## Historical Tracking

Track performance across multiple runs:

```python
from scripts.model_tracking import ModelPerformanceTracker

# Create tracker
tracker = ModelPerformanceTracker("model_performance_history.json")

# Load latest MLflow run
from scripts.model_tracking import load_mlflow_run_to_history
load_mlflow_run_to_history(run_id, "CDE", tracker)

# View summary
tracker.print_summary()

# Compare models
comparison_df = tracker.compare_models("cf_rmsre_full")
print(comparison_df)

# Export history
tracker.export_to_csv("model_performance_history.csv")
```


## Logged Metrics Reference

### For Each Model Type (CDE, SHRED):

**Training Metrics:**
- `{MODEL}_epochs` - Number of epochs (parameter)
- `{MODEL}_learning_rate` - Learning rate (parameter)
- `{MODEL}_batch_size` - Batch size (parameter)
- `{MODEL}_final_train_loss` - Final training loss
- `{MODEL}_final_valid_loss` - Final validation loss
- `{MODEL}_best_valid_loss` - Best validation loss during training
- `{MODEL}_n_epochs_trained` - Number of epochs actually trained
- `{MODEL}_train_loss_min/max/mean/std` - Loss statistics

**Architecture Metrics:**
- `{MODEL}_hidden_size` - Hidden layer size (parameter)
- `{MODEL}_total_parameters` - Total trainable parameters

**Evaluation Metrics:**
- `{MODEL}_cf_rmsre_full` - Cross-flow RMSRE on full sequence
- `{MODEL}_il_rmsre_full` - In-line RMSRE on full sequence
- `{MODEL}_cf_rmse_full` - Cross-flow RMSE on full sequence
- `{MODEL}_il_rmse_full` - In-line RMSE on full sequence
- `{MODEL}_cf_mae_full` - Cross-flow MAE on full sequence
- `{MODEL}_il_mae_full` - In-line MAE on full sequence
- `{MODEL}_cf_rmse_train/valid/test` - Split-wise CF metrics
- `{MODEL}_il_rmse_train/valid/test` - Split-wise IL metrics

**Comparison Metrics:**
- `CDE_vs_SHRED_cf_improvement_pct` - CDE % improvement over SHRED on CF
- `CDE_vs_SHRED_il_improvement_pct` - CDE % improvement over SHRED on IL


## MLflow Directory Structure

```
mlruns/
├── 0/                              # Experiment ID
│   └── meta.yaml
└── XXXXXXXX/                       # Your experiment with date-based ID
    ├── meta.yaml
    └── XXXXXXXX/                   # Run ID
        ├── meta.yaml
        ├── artifacts/              # Saved models, plots, etc.
        ├── metrics/                # Metric logs
        ├── params/                 # Parameter logs
        └── tags/
```


## Tips for Best Results

1. **Always log dataset info** - Required for reproducibility
2. **Use consistent experiment names** - Easier to compare across runs
3. **Add meaningful tags** - Helps filter and organize runs
4. **Log model checkpoints** - Can be added to artifacts/
5. **Review metrics regularly** - Use MLflow UI to spot trends

## Common Issues

**MLflow directory write errors:**
- Ensure you have write permissions in the workspace

**Missing metrics:**
- Check that all metrics are computed before logging
- Use float() to convert JAX arrays to Python floats

**Run not showing up:**
- MLflow creates a 'mlruns' directory locally
- Runs persist until explicitly deleted
- Use `mlflow.end_run()` to properly close runs

"""

# Quick start example
if __name__ == "__main__":
    print(__doc__)
    print("\nTo get started, run your notebook completely, then in a new cell:")
    print("\nfrom scripts.mlflow_analysis import summarize_experiment")
    print("summarize_experiment('VIV_NeuralCDE_vs_SHRED')")
