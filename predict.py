"""
TFT Baseflow Prediction Script
===============================
Downloads a trained TFT model from HuggingFace and runs predictions
on cluster-wise data using the exact same configs from training.

Usage:
    python predict.py --cluster 0
    python predict.py --cluster 0 --data path/to/custom_data.csv
    python predict.py --cluster 0 --model akshatshaw/tft-baseflow-c0-cv5fold-seq18-rmse185.0666-timestamp20260328_113728
    python predict.py --cluster 0 --output results/my_predictions.csv
    python predict.py --list-models
    python predict.py --all-clusters
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Default HF model repos (latest cv5fold-seq18 models) ──────────────────────
DEFAULT_MODELS = {
    0: "akshatshaw/tft-baseflow-c0-cv5fold-seq18-rmse185.0666-timestamp20260328_113728",
    1: "akshatshaw/tft-baseflow-c1-cv5fold-seq18-rmse665.7481-timestamp20260329_111459",
    2: "akshatshaw/tft-baseflow-c2-cv5fold-seq18-rmse317.5040-timestamp20260330_074905",
    3: "akshatshaw/tft-baseflow-c3-cv5fold-seq18-rmse90.7000-timestamp20260331_073611",
}

# ── Feature definitions (same as training notebook) ───────────────────────────
DYNAMIC_FEATURES = [
    "prcp(mm/day)", "tmax(C)", "tmin(C)", "tavg(C)",
    "srad_lw(w/m2)", "srad_sw(w/m2)", "wind_u(m/s)", "wind_v(m/s)",
    "wind(m/s)", "rel_hum(%)", "pet_gleam(mm/day)",
    "aet_gleam(mm/day)", "evap_canopy(mm/day)", "evap_surface(mm/day)",
    "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)", "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)",
]

STATIC_FEATURES = [
    "elev_mean", "slope_mean", "ghi_area", "dpsbar",
    "p_mean", "p_mean_anum", "high_prec_freq",
    "low_prec_freq", "max_high_prec_dur", "max_low_prec_dur",
    "aridity_p_pet", "water_frac", "trees_frac", "flooded_veg_frac",
    "crops_frac", "built_area_frac", "lai_mean", "soil_depth",
    "soil_awc_top", "soil_awc_sub", "soil_awsc_min", "soil_awsc_max",
    "sand_frac_top", "sand_frac_sub", "silt_frac_top", "silt_frac_sub",
    "clay_frac_top", "clay_frac_sub", "gravel_frac_top", "gravel_frac_sub",
    "bulkdense_top_mean", "bulkdens_sub_mean", "wtd", "org_carb_sub_mean",
    "org_carb_top_mean", "geol_porosity", "geol_permeability",
    "total_storage", "reservoir_index",
]

# ── Default data paths (relative to this script) ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = {
    c: os.path.join(SCRIPT_DIR, f"output/merged_clustered/cluster_{c}_combined_new.csv")
    for c in range(4)
}


def list_available_models():
    """List all TFT baseflow models on akshatshaw's HuggingFace."""
    api = HfApi()
    models = api.list_models(author="akshatshaw")
    tft_models = [m.id for m in models if "tft" in m.id.lower() or "baseflow" in m.id.lower()]
    print("Available TFT/baseflow models on HuggingFace:")
    for m in sorted(tft_models):
        marker = " (default)" if m in DEFAULT_MODELS.values() else ""
        print(f"  {m}{marker}")
    return tft_models


def download_model_and_config(repo_id: str):
    """Download model weights and model_card.json config from HuggingFace."""
    print(f"Downloading from {repo_id} ...")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")
    config_path = hf_hub_download(repo_id=repo_id, filename="model_card.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"  Model weights: {model_path}")
    print(f"  Config: {config['model_type']} | "
          f"encoder_length={config['architecture']['encoder_length']} | "
          f"prediction_length={config['architecture']['prediction_length']}")
    return model_path, config


def load_data(data_path: str, config: dict) -> pd.DataFrame:
    """Load and validate cluster data CSV."""
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Use features from model_card.json if available, else fall back to defaults
    static_feats = config.get("features", {}).get("static_features", STATIC_FEATURES)
    dynamic_feats = config.get("features", {}).get("dynamic_features", DYNAMIC_FEATURES)
    all_required = static_feats + dynamic_feats + ["baseflow", "month", "day", "year"]

    missing = [c for c in all_required if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        sys.exit(1)

    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values with column means ...")
        df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


def build_dataset(df: pd.DataFrame, config: dict) -> TimeSeriesDataSet:
    """Build pytorch_forecasting TimeSeriesDataSet matching the training config."""
    arch = config["architecture"]
    static_feats = config.get("features", {}).get("static_features", STATIC_FEATURES)
    dynamic_feats = config.get("features", {}).get("dynamic_features", DYNAMIC_FEATURES)

    df_tft = df.copy()
    df_tft["time_idx"] = range(len(df_tft))
    df_tft["group"] = 0

    for col in ["month", "day", "year"]:
        df_tft[col] = df_tft[col].astype(str)

    if "cluster" in df_tft.columns:
        df_tft.drop(columns=["cluster"], inplace=True)
    if "gauge_id" in df_tft.columns:
        df_tft.drop(columns=["gauge_id"], inplace=True)

    max_encoder_length = arch["encoder_length"]
    max_prediction_length = arch["prediction_length"]

    dataset = TimeSeriesDataSet(
        df_tft,
        time_idx="time_idx",
        target="baseflow",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=static_feats,
        time_varying_known_reals=dynamic_feats,
        time_varying_known_categoricals=["month", "day", "year"],
        time_varying_unknown_reals=["baseflow"],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def load_model(model_path: str, config: dict, dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Instantiate TFT from config + dataset, then load HF weights."""
    arch = config["architecture"]

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=config["training"]["learning_rate"],
        hidden_size=arch["hidden_size"],
        attention_head_size=arch["attention_head_size"],
        dropout=arch["dropout"],
        hidden_continuous_size=arch["hidden_continuous_size"],
        output_size=1,
        loss=RMSE(),
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params / 1e3:.1f}k parameters")
    return model


def calculate_kge(observed, predicted):
    """Kling-Gupta Efficiency."""
    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(observed)
    beta = np.mean(predicted) / np.mean(observed)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def predict(model, dataset, batch_size=1024):
    """Run predictions, return (y_true, y_pred)."""
    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    predictions = model.predict(dataloader, return_y=True)
    y_pred = predictions.output.cpu().numpy().flatten()
    y_true = predictions.y[0].cpu().numpy().flatten()
    return y_true, y_pred


def evaluate(y_true, y_pred):
    """Compute and print metrics."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    kge = calculate_kge(y_true, y_pred)

    print("\n-- Evaluation Metrics --")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  KGE  : {kge:.4f}")
    print(f"  Samples: {len(y_true)}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "kge": kge}


def save_predictions(y_true, y_pred, output_path: str):
    """Save predictions CSV."""
    df_out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def run_single_cluster(cluster, repo_id, data_path, output_path, batch_size, no_eval):
    """Full pipeline for one cluster."""
    print("=" * 60)
    print(f"TFT Baseflow Prediction — Cluster {cluster}")
    print("=" * 60)
    print(f"  Model : {repo_id}")
    print(f"  Data  : {data_path}")
    print(f"  Output: {output_path}")
    print()

    model_path, config = download_model_and_config(repo_id)
    df = load_data(data_path, config)

    print("Building TimeSeriesDataSet ...")
    dataset = build_dataset(df, config)
    print(f"  Dataset samples: {len(dataset)}")

    print("Loading model ...")
    model = load_model(model_path, config, dataset)

    print("Running predictions ...")
    y_true, y_pred = predict(model, dataset, batch_size=batch_size)

    metrics = None
    if not no_eval:
        metrics = evaluate(y_true, y_pred)

    save_predictions(y_true, y_pred, output_path)
    print("\nDone.")
    return y_true, y_pred, metrics


def main():
    parser = argparse.ArgumentParser(
        description="TFT Baseflow Prediction — download model from HF and predict on cluster data"
    )
    parser.add_argument(
        "--cluster", "-c", type=int, choices=[0, 1, 2, 3],
        help="Cluster ID (0-3). Uses default HF model and local data for that cluster."
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="HuggingFace repo ID override (e.g. akshatshaw/tft-baseflow-c0-...)."
    )
    parser.add_argument(
        "--data", "-d", type=str, default=None,
        help="Path to input CSV data. Overrides default for --cluster."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path for output predictions CSV."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1024,
        help="Batch size for prediction (default: 1024)"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models on HuggingFace and exit."
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation metrics."
    )
    parser.add_argument(
        "--all-clusters", action="store_true",
        help="Run prediction on all 4 clusters sequentially."
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    if args.all_clusters:
        print("=" * 60)
        print("Batch Prediction — All Clusters")
        print("=" * 60)
        for c in range(4):
            repo = DEFAULT_MODELS[c]
            data = DEFAULT_DATA[c]
            out = os.path.join(SCRIPT_DIR, f"output/predictions/pred_c{c}.csv")
            try:
                run_single_cluster(c, repo, data, out, args.batch_size, args.no_eval)
            except Exception as e:
                print(f"\nC{c}: FAILED — {e}")
            print()
        return

    if args.cluster is None and args.model is None:
        parser.error("Provide --cluster (0-3), --model <hf_repo_id>, or --all-clusters")

    repo_id = args.model if args.model else DEFAULT_MODELS[args.cluster]

    if args.data:
        data_path = args.data
    elif args.cluster is not None:
        data_path = DEFAULT_DATA[args.cluster]
    else:
        parser.error("Provide --data when using --model without --cluster")

    cluster_label = args.cluster if args.cluster is not None else "custom"
    output_path = args.output or os.path.join(
        SCRIPT_DIR, f"output/predictions/pred_c{cluster_label}.csv"
    )

    run_single_cluster(cluster_label, repo_id, data_path, output_path, args.batch_size, args.no_eval)


if __name__ == "__main__":
    main()
