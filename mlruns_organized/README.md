# Organized MLflow Runs

This directory contains organized results from notebook executions with intuitive, human-readable naming.

## Structure

```
mlruns_organized/
├── test_vivdata_diffrax/
│   ├── 2026-03-04_v1/
│   │   ├── metrics.json          # All model performance metrics (RMSE for all models/splits)
│   │   ├── params.json           # Training hyperparameters (batch_size, epochs, lag, stride)
│   │   ├── dataset_info.json     # Dataset configuration (train/valid/test split, sensors, POD modes)
│   │   └── model_info.json       # Model architectures and parameter counts
│   ├── 2026-03-04_v2/
│   └── ...
└── vivdata/
    ├── 2026-03-04_v1/
    │   ├── metrics.json          # POD analysis metrics and singular values
    │   ├── params.json           # Analysis parameters
    │   ├── dataset_info.json     # Dataset configuration
    │   └── model_info.json       # POD methodology
    └── ...
```

## File Descriptions

### metrics.json
Contains all performance metrics from the run.

**For test_vivdata_diffrax:**
- RMSE (%) for each model (NeuralCDE, SHRED_LSTM, MLP, SHREDAttention)
- Split: train, valid, test
- Directions: CF (Cross-Flow) and IL (Inline)

Example:
```json
{
  "NeuralCDE": {
    "num_parameters": 15234,
    "train_cf_rmse": 2.45,
    "train_il_rmse": 2.89,
    "test_cf_rmse": 3.12,
    "test_il_rmse": 3.45
  },
  ...
}
```

**For vivdata:**
- POD singular values (first 5)
- Energy explained (%)
- Number of modes extracted

### params.json
Training/analysis hyperparameters.

**For test_vivdata_diffrax:**
- batch_size, epochs, lag, stride

**For vivdata:**
- decomposition_method, dataset_source, output_format

### dataset_info.json
Dataset configuration details.

```json
{
  "split": {
    "train_samples": 1600,
    "valid_samples": 200,
    "test_samples": 200,
    "total_samples": 2000
  },
  "sensors": {
    "num_sensors": 5,
    "cf_modes": 10,
    "il_modes": 12
  }
}
```

### model_info.json
Model architecture and parameter information.

```json
{
  "NeuralCDE": {
    "name": "NeuralCDE",
    "num_parameters": 15234,
    "architecture": null
  },
  ...
}
```

## Usage

The notebooks automatically save results to organized folders when executed. Each run gets:
- **Date folder**: `YYYY-MM-DD` (today's date)
- **Version number**: `v1`, `v2`, `v3`, etc. (auto-incrementing)
- **Combined name**: `2026-03-04_v1`

### Example: My First Run

1. Run `test_vivdata_diffrax.ipynb` on March 4, 2026
2. Results saved to: `mlruns_organized/test_vivdata_diffrax/2026-03-04_v1/`
3. Run again same day
4. Results saved to: `mlruns_organized/test_vivdata_diffrax/2026-03-04_v2/`

### Viewing Results

To view results programmatically:

```python
import json
from pathlib import Path

# Load metrics from a run
metrics_file = Path("mlruns_organized/test_vivdata_diffrax/2026-03-04_v1/metrics.json")
with open(metrics_file) as f:
    metrics = json.load(f)

print(metrics["NeuralCDE"]["test_cf_rmse"])  # Get specific metric
```

### Comparison Script

To compare runs across different dates/versions:

```python
from pathlib import Path
import json

runs_dir = Path("mlruns_organized/test_vivdata_diffrax")
results = {}

for run_folder in sorted(runs_dir.iterdir()):
    metrics_file = run_folder / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            results[run_folder.name] = json.load(f)

# Compare NeuralCDE test RMSE across runs
for run_name, metrics in results.items():
    rmse = metrics["NeuralCDE"]["test_cf_rmse"]
    print(f"{run_name}: CF RMSE = {rmse:.2f}%")
```

## Integration with MLflow

These organized folders **complement** the MLflow tracking (which still uses cryptic UUIDs internally). 
- **MLflow UI**: Go to `http://localhost:5000` to see pretty dashboards (run with `mlflow ui`)
- **Organized folders**: Use these for programmatic access and easy version comparison

## Notebook Integration

The utility is used in both notebooks:

**File**: `scripts/mlruns_utils.py`

```python
from scripts.mlruns_utils import MLRunOrganizer

organizer = MLRunOrganizer(notebook_name="test_vivdata_diffrax")
run_dir = organizer.save_all(
    metrics=metrics,
    params=params,
    dataset_info=dataset_info,
    model_info=model_info
)
```

Version numbering is automatic - each notebook run increments the version for the current date.
