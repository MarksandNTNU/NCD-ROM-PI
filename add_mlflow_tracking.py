"""Script to add MLflow tracking cells to the pi_cde notebook."""

import json
from pathlib import Path

notebook_path = Path('/Users/markussandnes/Desktop/NCD-ROM-new/notebooks/pi_cde.ipynb')

# Read the notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Update the imports in the first code cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('import numpy' in line for line in cell.get('source', [])):
        cell['source'] = [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from matplotlib import animation\n",
            "from IPython.display import HTML\n",
            "import mlflow\n",
            "import mlflow.pytorch\n",
            "from datetime import datetime\n",
            "from pathlib import Path\n",
            "from scripts.mlflow_utils import start_mlflow_run, log_cde_model_params, log_training_params"
        ]
        print(f"✓ Updated imports in cell {i}")
        break

# 2. Find the cell with model instantiation (contains "hidden_size = ") and add MLflow init after it
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('decoder_sizes = [64, 128]' in line for line in cell.get('source', [])):
        # Insert a markdown cell and then a code cell with MLflow initialization
        # First, find the next cell position
        mlflow_markdown_cell = {
            "cell_type": "markdown",
            "id": "mlflow-md-1",
            "metadata": {},
            "source": [
                "## MLflow Experiment Tracking\n",
                "This section initializes MLflow to track all model parameters, hyperparameters, and evaluation metrics for reproducibility and comparison."
            ]
        }
        
        mlflow_init_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "mlflow-init",
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Initialize MLflow tracking ──────────────────────────────────────\n",
                "experiment_name = \"PI_CDE_Transport_Equation\"\n",
                "project_root = Path.cwd()\n",
                "\n",
                "run_name, run_id = start_mlflow_run(\n",
                "    experiment_name=experiment_name,\n",
                "    project_root=project_root,\n",
                "    tags={\"model_type\": \"NeuralCDE\", \"equation\": \"transport\"}\n",
                ")\n",
                "print(f\"✓ MLflow Run: {run_name} ({run_id[:8]}…)\")\n",
                "\n",
                "# ── Log dataset parameters ──────────────────────────────────────────\n",
                "mlflow.log_params({\n",
                "    \"data_equation\": \"transport_equation\",\n",
                "    \"lag\": lag,\n",
                "    \"data_size\": data_size,\n",
                "    \"n_spatial_points\": Nx,\n",
                "    \"n_temporal_points\": Nt,\n",
                "    \"n_ics\": N_ics,\n",
                "    \"n_samples_total\": N_samples,\n",
                "    \"train_set_size\": n_train,\n",
                "    \"valid_set_size\": n_valid,\n",
                "    \"test_set_size\": len(test_idx),\n",
                "    \"n_pod_modes\": n_modes,\n",
                "    \"pod_train_rel_error\": f\"{rel_err_train:.6f}\",\n",
                "    \"pod_valid_rel_error\": f\"{rel_err_valid:.6f}\",\n",
                "    \"pod_test_rel_error\": f\"{rel_err_test:.6f}\",\n",
                "})\n",
                "print(f\"✓ Dataset params logged: {n_train} train, {n_valid} valid, {len(test_idx)} test\")\n",
                "\n",
                "# ── Log NeuralCDE model architecture parameters ──────────────────────\n",
                "log_cde_model_params(\n",
                "    hidden_size=hidden_size,\n",
                "    width_size=width_size,\n",
                "    depth=depth,\n",
                "    output_size=output_size,\n",
                "    decoder_sizes=decoder_sizes,\n",
                ")\n",
                "mlflow.log_param(\"total_parameters\", n_params)\n",
                "print(f\"✓ Model architecture logged: {n_params:,} total parameters\")"
            ]
        }
        
        nb['cells'].insert(i + 1, mlflow_markdown_cell)
        nb['cells'].insert(i + 2, mlflow_init_cell)
        print(f"✓ Inserted MLflow initialization cells after cell {i}")
        break

# 3. Add training config and metrics logging cells
cells_added = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('early_stopping=30' in line for line in cell.get('source', [])):
        if 'fit_CDE' not in ''.join(cell.get('source', [])):
            continue
        # Check if this is the first training call
        if 'train_losses, valid_losses = fit_CDE' in ''.join(cell.get('source', [])):
            # Add training config before this cell
            training_config_cell = {
                "cell_type": "code",
                "execution_count": None,
                "id": "mlflow-training-config",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ── Define training learning rates ──────────────────────────────────\n",
                    "lr_1 = 1e-3\n",
                    "lr_2 = 1e-4\n",
                    "\n",
                    "# ── Log initial training hyperparameters ────────────────────────\n",
                    "log_training_params(\n",
                    "    epochs=epochs,\n",
                    "    learning_rate=lr_1,\n",
                    "    batch_size=batch_size,\n",
                    "    patience=30,\n",
                    "    model_type=\"CDE\",\n",
                    ")\n",
                    "print(f\"✓ Training config logged: {epochs} epochs, batch_size={batch_size}, initial_lr={lr_1}\")"
                ]
            }
            nb['cells'].insert(i, training_config_cell)
            print(f"✓ Inserted training config cell at {i}")
            cells_added += 1
            i += 1  # Adjust index
        
        # After second training call, add metrics logging
        elif 'train_losses_2, valid_losses_2' in ''.join(cell.get('source', [])):
            metrics_logging_cell = {
                "cell_type": "code",
                "execution_count": None,
                "id": "mlflow-training-metrics",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ── Log combined training metrics ────────────────────────────────────\n",
                    "combined_train_losses = train_losses + train_losses_2\n",
                    "combined_valid_losses = valid_losses + valid_losses_2\n",
                    "\n",
                    "mlflow.log_metrics({\n",
                    "    \"initial_train_loss\": float(train_losses[0]),\n",
                    "    \"initial_valid_loss\": float(valid_losses[0]),\n",
                    "    \"final_train_loss_stage1\": float(train_losses[-1]),\n",
                    "    \"final_valid_loss_stage1\": float(valid_losses[-1]),\n",
                    "    \"best_valid_loss_stage1\": float(min(valid_losses)),\n",
                    "    \"final_train_loss_stage2\": float(train_losses_2[-1]),\n",
                    "    \"final_valid_loss_stage2\": float(valid_losses_2[-1]),\n",
                    "    \"best_valid_loss_stage2\": float(min(valid_losses_2)),\n",
                    "    \"best_valid_loss_combined\": float(min(combined_valid_losses)),\n",
                    "    \"epochs_stage1\": len(train_losses),\n",
                    "    \"epochs_stage2\": len(train_losses_2),\n",
                    "    \"total_epochs\": len(combined_train_losses),\n",
                    "}, step=0)\n",
                    "\n",
                    "print(f\"\\n✓ Training metrics logged:\")\n",
                    "print(f\"  Stage 1 (lr={lr_1}): Best valid loss = {min(valid_losses):.6e}\")\n",
                    "print(f\"  Stage 2 (lr={lr_2}): Best valid loss = {min(valid_losses_2):.6e}\")\n",
                    "print(f\"  Combined: Best valid loss = {min(combined_valid_losses):.6e}\")"
                ]
            }
            nb['cells'].insert(i + 1, metrics_logging_cell)
            print(f"✓ Inserted training metrics logging cell at {i + 1}")
            cells_added += 1
            break

# 4. Add evaluation metrics logging (after model predictions on train set)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('preds_field_train' in line and 'mse_train' in ''.join(cell.get('source', [])) for line in cell.get('source', [])):
        eval_logging_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "mlflow-eval-metrics",
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Log evaluation metrics on train set ──────────────────────────────\n",
                "mlflow.log_metrics({\n",
                "    \"train_mse_full_field\": mse_train,\n",
                "    \"train_relative_l2_error\": rel_l2 * 100,  # as percentage\n",
                "}, step=1)\n",
                "\n",
                "print(f\"✓ Train evaluation metrics logged:\")\n",
                "print(f\"  MSE: {mse_train:.6e}\")\n",
                "print(f\"  Relative L2: {rel_l2*100:.4f}%\")"
            ]
        }
        nb['cells'].insert(i + 1, eval_logging_cell)
        print(f"✓ Inserted evaluation metrics logging cell at {i + 1}")
        break

# 5. Add physics residual metrics logging
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('TRANSPORT EQUATION RESIDUAL' in line for line in cell.get('source', [])):
        physics_logging_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "mlflow-physics-metrics",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compute additional error metrics for logging\n",
                "err_u = np.abs(U_true - u_pred[which])\n",
                "err_dudt = np.abs(dU_dt_true - du_dt_pred[which])\n",
                "err_dudx = np.abs(dU_dx_true - du_dx_pred[which])\n",
                "\n",
                "# ── Log physics residual metrics ─────────────────────────────────────\n",
                "mlflow.log_metrics({\n",
                "    \"residual_l2_norm\": float(res_norm),\n",
                "    \"du_dt_l2_norm\": float(dudt_norm),\n",
                "    \"du_dx_l2_norm\": float(dudx_norm),\n",
                "    \"mean_field_error\": float(np.mean(err_u)),\n",
                "    \"max_field_error\": float(np.max(err_u)),\n",
                "    \"mean_dudt_error\": float(np.mean(err_dudt)),\n",
                "    \"max_dudt_error\": float(np.max(err_dudt)),\n",
                "    \"mean_dudx_error\": float(np.mean(err_dudx)),\n",
                "    \"max_dudx_error\": float(np.max(err_dudx)),\n",
                "    \"wave_speed_c\": float(c),\n",
                "}, step=2)\n",
                "\n",
                "print(f\"\\n✓ Physics residual metrics logged:\")\n",
                "print(f\"  Residual norm: {res_norm:.6e}\")\n",
                "print(f\"  Mean field error: {np.mean(err_u):.6e}\")\n",
                "print(f\"  Mean ∂u/∂t error: {np.mean(err_dudt):.6e}\")\n",
                "print(f\"  Mean ∂u/∂x error: {np.mean(err_dudx):.6e}\")"
            ]
        }
        nb['cells'].insert(i + 1, physics_logging_cell)
        print(f"✓ Inserted physics metrics logging cell at {i + 1}")
        break

# 6. Add final MLflow run closure
final_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "mlflow-close-run",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Log final test metrics and close MLflow run ──────────────────────\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"MLflow Run Summary\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Log final metrics on full dataset\n",
        "mlflow.log_metrics({\n",
        "    \"full_dataset_mse\": float(mse_full),\n",
        "    \"full_dataset_relative_l2_error\": float(rel_l2 * 100),\n",
        "    \"full_dataset_rmse\": float(np.sqrt(mse_full)),\n",
        "}, step=3)\n",
        "\n",
        "# Log metadata\n",
        "mlflow.log_param(\"notebook_name\", \"pi_cde.ipynb\")\n",
        "mlflow.log_param(\"experiment_completion_time\", datetime.now().isoformat())\n",
        "\n",
        "# End the run\n",
        "mlflow.end_run()\n",
        "\n",
        "print(f\"\\n✓ MLflow run completed successfully!\")\n",
        "print(f\"  Run name: {run_name}\")\n",
        "print(f\"  Run ID:   {run_id}\")\n",
        "print(f\"\\nTo view results:\")\n",
        "print(f\"  → From terminal: mlflow ui\")\n",
        "print(f\"  → Open browser: http://localhost:5000\")\n",
        "print(\"=\"*70)"
    ]
}
nb['cells'].append(final_cell)
print(f"✓ Appended final MLflow cell closure")

# Write back the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n✅ MLflow tracking successfully added to pi_cde.ipynb!")
print(f"Total cells in notebook: {len(nb['cells'])}")
