"""Model performance tracking and history utilities."""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd


class ModelPerformanceTracker:
    """Track and compare model performance over time."""
    
    def __init__(self, history_file: str = "model_performance_history.json"):
        """
        Initialize tracker.
        
        Args:
            history_file: Path to JSON file storing performance history
        """
        self.history_file = Path(history_file)
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load existing history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self) -> None:
        """Save history to file."""
        # Ensure parent directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def add_model_performance(
        self,
        model_type: str,
        run_id: str,
        metrics: Dict,
        params: Dict,
        timestamp: str = None,
    ) -> None:
        """
        Add a model's performance to history.
        
        Args:
            model_type: Type of model (e.g., "CDE", "SHRED")
            run_id: MLflow run ID
            metrics: Dictionary of metrics
            params: Dictionary of parameters
            timestamp: Optional timestamp (auto-generated if not provided)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "model_type": model_type,
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
        }
        
        self.history.append(entry)
        self._save_history()
        print(f"Added {model_type} performance record (Run: {run_id[:8]}...)")
    
    def get_model_history(self, model_type: str = None) -> List[Dict]:
        """
        Get history for a specific model type.
        
        Args:
            model_type: Type of model to filter by. If None, return all.
            
        Returns:
            List of history entries
        """
        if model_type is None:
            return self.history
        
        return [entry for entry in self.history if entry["model_type"] == model_type]
    
    def get_best_model(
        self,
        model_type: str,
        metric: str,
        mode: str = "min",  # 'min' for error, 'max' for accuracy
    ) -> Dict:
        """
        Get best model by a specific metric.
        
        Args:
            model_type: Type of model
            metric: Name of metric to optimize
            mode: 'min' or 'max'
            
        Returns:
            Best model entry
        """
        entries = self.get_model_history(model_type)
        
        if not entries:
            return None
        
        # Extract values for comparison
        valid_entries = [e for e in entries if metric in e.get("metrics", {})]
        
        if not valid_entries:
            return None
        
        if mode == "min":
            best = min(valid_entries, key=lambda e: e["metrics"][metric])
        else:
            best = max(valid_entries, key=lambda e: e["metrics"][metric])
        
        return best
    
    def compare_models(self, metric: str) -> pd.DataFrame:
        """
        Create comparison DataFrame for all models on a specific metric.
        
        Args:
            metric: Name of metric to compare
            
        Returns:
            DataFrame with comparison
        """
        data = []
        
        for entry in self.history:
            if metric in entry.get("metrics", {}):
                row = {
                    "timestamp": entry["timestamp"],
                    "model_type": entry["model_type"],
                    "run_id": entry["run_id"][:8],
                    metric: entry["metrics"][metric],
                }
                # Add selected parameters
                for param_name in ["hidden_size", "batch_size", "learning_rate"]:
                    if param_name in entry.get("params", {}):
                        row[param_name] = entry["params"][param_name]
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def print_summary(self) -> None:
        """Print summary of all tracked models."""
        if not self.history:
            print("No model history recorded yet")
            return
        
        print(f"\n{'='*80}")
        print(f"Model Performance Summary")
        print(f"{'='*80}\n")
        
        # Group by model type
        model_types = set(entry["model_type"] for entry in self.history)
        
        for model_type in sorted(model_types):
            entries = self.get_model_history(model_type)
            print(f"{model_type}: {len(entries)} runs recorded")
            
            # Show latest
            latest = entries[-1]
            print(f"  Latest Run: {latest['run_id'][:8]}... ({latest['timestamp']})")
            print(f"  Key Metrics:")
            
            for metric_name, metric_value in latest["metrics"].items():
                if "rmsre" in metric_name or "rmse" in metric_name:
                    print(f"    {metric_name}: {metric_value:.6f}")
            
            print()
    
    def export_to_csv(self, output_path: str = "model_performance_history.csv") -> None:
        """Export complete history to CSV."""
        if not self.history:
            print("No history to export")
            return
        
        # Flatten history for CSV export
        rows = []
        for entry in self.history:
            row = {
                "timestamp": entry["timestamp"],
                "model_type": entry["model_type"],
                "run_id": entry["run_id"],
            }
            
            # Add metrics with prefixes
            for metric_name, metric_value in entry.get("metrics", {}).items():
                row[f"metric_{metric_name}"] = metric_value
            
            # Add params with prefixes
            for param_name, param_value in entry.get("params", {}).items():
                row[f"param_{param_name}"] = param_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"History exported to {output_path}")


def load_mlflow_run_to_history(
    run_id: str,
    model_type: str,
    tracker: ModelPerformanceTracker,
) -> None:
    """
    Load a specific MLflow run into the history tracker.
    
    Args:
        run_id: MLflow run ID
        model_type: Type of model
        tracker: ModelPerformanceTracker instance
    """
    import mlflow
    
    try:
        run = mlflow.get_run(run_id)
        
        metrics = run.data.metrics
        params = run.data.params
        timestamp = run.info.start_time
        
        # Convert timestamp to ISO format if needed
        if isinstance(timestamp, int):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()
        
        tracker.add_model_performance(
            model_type=model_type,
            run_id=run_id,
            metrics=metrics,
            params=params,
            timestamp=timestamp,
        )
        
    except Exception as e:
        print(f"Error loading run {run_id}: {e}")
