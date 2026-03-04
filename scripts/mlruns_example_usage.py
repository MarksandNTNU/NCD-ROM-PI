"""
Example of how to use MLRunOrganizer in notebooks.

This demonstrates how to integrate the organizer into your Jupyter notebooks.
"""

# In your notebook, at the END after all training/evaluation:

from scripts.mlruns_utils import MLRunOrganizer, create_dataset_info, create_model_info

# Initialize the organizer with your notebook name
organizer = MLRunOrganizer(notebook_name="test_vivdata_diffrax")

# Prepare dataset information
dataset_info = create_dataset_info(
    n_train=n_train,
    n_valid=n_valid,
    n_test=n_test,
    n_sensors=nsensors,
    modes_cf=modes_cf,
    modes_il=modes_il
)

# Prepare metrics for all models
metrics = {
    "NeuralCDE": {
        "cf_rmse_train": float(rmsre(preds_cde_cf_train, Y_cf_train) * 100),
        "cf_rmse_valid": float(rmsre(preds_cde_cf_valid, Y_cf_valid) * 100),
        "cf_rmse_test": float(rmsre(preds_cde_cf_test, Y_cf_test) * 100),
        "il_rmse_train": float(rmsre(preds_cde_il_train, Y_il_train) * 100),
        "il_rmse_valid": float(rmsre(preds_cde_il_valid, Y_il_valid) * 100),
        "il_rmse_test": float(rmsre(preds_cde_il_test, Y_il_test) * 100),
    },
    "SHRED_LSTM": {
        "cf_rmse_train": float(shred_rmse_cf_train),
        "cf_rmse_valid": float(shred_rmse_cf_valid),
        "cf_rmse_test": float(shred_rmse_cf_test),
        "il_rmse_train": float(shred_rmse_il_train),
        "il_rmse_valid": float(shred_rmse_il_valid),
        "il_rmse_test": float(shred_rmse_il_test),
    },
    "MLP": {
        "cf_rmse_train": float(mlp_rmse_cf_train),
        "cf_rmse_valid": float(mlp_rmse_cf_valid),
        "cf_rmse_test": float(mlp_rmse_cf_test),
        "il_rmse_train": float(mlp_rmse_il_train),
        "il_rmse_valid": float(mlp_rmse_il_valid),
        "il_rmse_test": float(mlp_rmse_il_test),
    },
    "SHREDAttention": {
        "cf_rmse_train": float(attn_rmse_cf_train),
        "cf_rmse_valid": float(attn_rmse_cf_valid),
        "cf_rmse_test": float(attn_rmse_cf_test),
        "il_rmse_train": float(attn_rmse_il_train),
        "il_rmse_valid": float(attn_rmse_il_valid),
        "il_rmse_test": float(attn_rmse_il_test),
    },
}

# Prepare model information
model_info = {
    "NeuralCDE": create_model_info("NeuralCDE", cde_params),
    "SHRED_LSTM": create_model_info("SHRED_LSTM", shred_params),
    "MLP": create_model_info("MLP", mlp_params),
    "SHREDAttention": create_model_info("SHREDAttention", attn_params),
}

# Prepare parameters
params = {
    "batch_size": batch_size,
    "epochs": epochs,
    "lag": lag,
    "stride": stride,
}

# Save everything to organized folder
run_dir = organizer.save_all(
    metrics=metrics,
    params=params,
    dataset_info=dataset_info,
    model_info=model_info
)

print(f"\n✓ Run saved to: {run_dir}")
