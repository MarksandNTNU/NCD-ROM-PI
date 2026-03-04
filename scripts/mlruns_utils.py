"""
Utility module for organizing MLflow runs with intuitive folder structure.

Structure:
mlruns_organized/
├── test_vivdata_diffrax/
│   ├── 2026-03-04_v1/
│   │   ├── metrics.json
│   │   ├── params.json
│   │   ├── dataset_info.json
│   │   └── model_info.json
│   └── ...
└── vivdata/
    └── ...
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MLRunOrganizer:
    """Manages organized storage of MLflow runs with intuitive naming."""
    
    def __init__(self, notebook_name: str, base_dir: str = "mlruns_organized"):
        """
        Initialize the organizer.
        
        Args:
            notebook_name: Name of the notebook (e.g., 'test_vivdata_diffrax', 'vivdata')
            base_dir: Base directory for organized runs (default: 'mlruns_organized')
        """
        self.notebook_name = notebook_name
        self.base_dir = Path(base_dir)
        self.notebook_dir = self.base_dir / notebook_name
        self.today = datetime.now().strftime("%Y-%m-%d")
        
    def get_next_version(self) -> int:
        """Get the next version number for today's runs."""
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all existing folders for today
        today_folders = [
            d for d in self.notebook_dir.iterdir()
            if d.is_dir() and d.name.startswith(self.today)
        ]
        
        if not today_folders:
            return 1
        
        # Extract version numbers and get the max
        versions = []
        for folder in today_folders:
            try:
                version = int(folder.name.split("_v")[-1])
                versions.append(version)
            except (ValueError, IndexError):
                pass
        
        return max(versions) + 1 if versions else 1
    
    def create_run_folder(self) -> Path:
        """Create and return the run folder path for today."""
        version = self.get_next_version()
        run_name = f"{self.today}_v{version}"
        run_dir = self.notebook_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def save_metrics(self, metrics: Dict[str, Any], run_dir: Path) -> None:
        """Save metrics to metrics.json."""
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_params(self, params: Dict[str, Any], run_dir: Path) -> None:
        """Save parameters to params.json."""
        params_file = run_dir / "params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
    
    def save_dataset_info(self, dataset_info: Dict[str, Any], run_dir: Path) -> None:
        """Save dataset information to dataset_info.json."""
        dataset_file = run_dir / "dataset_info.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def save_model_info(self, model_info: Dict[str, Any], run_dir: Path) -> None:
        """Save model information to model_info.json."""
        model_file = run_dir / "model_info.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def save_all(self, metrics: Dict = None, params: Dict = None, 
                 dataset_info: Dict = None, model_info: Dict = None) -> Path:
        """
        Save all information in one call.
        
        Returns:
            Path to the created run directory
        """
        run_dir = self.create_run_folder()
        
        if metrics:
            self.save_metrics(metrics, run_dir)
        if params:
            self.save_params(params, run_dir)
        if dataset_info:
            self.save_dataset_info(dataset_info, run_dir)
        if model_info:
            self.save_model_info(model_info, run_dir)
        
        return run_dir


def create_test_vivdata_metrics(models_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create comprehensive metrics dictionary for test_vivdata_diffrax.
    
    Args:
        models_results: Dict with model names as keys and their metrics as values
        
    Returns:
        Formatted metrics dictionary
    """
    return {
        "models_evaluated": list(models_results.keys()),
        "evaluation_timestamp": datetime.now().isoformat(),
        "models": models_results
    }


def create_dataset_info(n_train: int, n_valid: int, n_test: int, 
                        n_sensors: int, modes_cf: int, modes_il: int,
                        cf_shape: tuple = None, il_shape: tuple = None) -> Dict[str, Any]:
    """Create standardized dataset information dictionary."""
    return {
        "split": {
            "train_samples": n_train,
            "valid_samples": n_valid,
            "test_samples": n_test,
            "total_samples": n_train + n_valid + n_test,
        },
        "sensors": {
            "num_sensors": n_sensors,
            "cf_modes": modes_cf,
            "il_modes": modes_il,
        },
        "shapes": {
            "cf_shape": str(cf_shape) if cf_shape else None,
            "il_shape": str(il_shape) if il_shape else None,
        },
        "created_at": datetime.now().isoformat(),
    }


def create_model_info(model_name: str, num_params: int, 
                      architecture: str = None, other_info: Dict = None) -> Dict[str, Any]:
    """Create standardized model information dictionary."""
    info = {
        "name": model_name,
        "num_parameters": num_params,
        "architecture": architecture,
    }
    if other_info:
        info.update(other_info)
    return info
