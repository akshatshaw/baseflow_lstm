# Cluster-Guided Deep Learning for Baseflow Prediction in Indian River Basins Using CAMELS-IND

---

## 1. Project Overview

### Problem Statement

Baseflow -- the portion of streamflow sustained by groundwater discharge -- is a critical hydrological variable for water resource management, drought assessment, and ecological flow estimation. Predicting baseflow at ungauged or sparsely gauged catchments across India remains a significant challenge due to the heterogeneous hydroclimatic and physiographic landscape of the subcontinent.

### Motivation

India's river systems span diverse climatic zones, geological formations, and land-use patterns. Traditional lumped hydrological models often fail to generalize across such heterogeneity. This project investigates whether **clustering catchments by static attributes** and subsequently training **cluster-specific deep learning models** can yield more accurate and interpretable baseflow predictions compared to a single global model.

### Real-World Relevance

- Water resource planning for 242 catchments across India
- Baseflow estimation supports irrigation scheduling, reservoir operations, and environmental flow requirements
- Regional clustering enables transfer learning to ungauged basins within the same hydroclimatic cluster

### Task Type

- **Primary task**: Time-series regression (daily baseflow prediction)
- **Supporting task**: Unsupervised clustering of catchments by static attributes

---

## 2. Dataset

### Source

The project uses the **CAMELS-IND** (Catchment Attributes and MEteorology for Large-sample Studies -- India) dataset, a large-sample hydrology dataset analogous to CAMELS-US, CAMELS-GB, and CAMELS-CL.

### Data Composition

**Static Attributes (211 features per catchment):**

| Category | Example Features | Count |
|----------|-----------------|-------|
| Topography | `elev_mean`, `slope_mean`, `dpsbar` | ~5 |
| Climate | `p_mean`, `aridity_p_pet`, `high_prec_freq`, `low_prec_freq` | ~12 |
| Hydrology | `bfi`, `runoff_ratio`, `q_mean`, `mean_anum_flow`, `freq_q_low`, `freq_q_high` | ~15 |
| Soil | `soil_depth`, `soil_awc_top`, `soil_awc_sub`, `sand_frac_top`, `clay_frac_top`, `bulkdense_top_mean` | ~18 |
| Geology | `geol_porosity`, `geol_permeability`, `geol_class_1st`, `geol_class_2nd` | ~5 |
| Land Cover | `crops_frac`, `trees_frac`, `water_frac`, `built_area_frac`, `lai_mean` | ~10 |
| Anthropogenic | `num_dams`, `reservoir_index`, `irrigation_frac`, `pop_density_2020` | ~5 |

**Dynamic Forcing Features (18 time-varying features, daily resolution):**

| Feature | Unit |
|---------|------|
| `prcp` | mm/day |
| `tmax`, `tmin`, `tavg` | Celsius |
| `srad_lw`, `srad_sw` | W/m^2 |
| `wind_u`, `wind_v`, `wind` | m/s |
| `rel_hum` | % |
| `pet_gleam`, `aet_gleam` | mm/day |
| `evap_canopy`, `evap_surface` | mm/day |
| `sm_lvl1` through `sm_lvl4` | kg/m^2 |

**Target Variable:** Baseflow (mm/day), extracted via the Boughton baseflow separation method applied to observed streamflow records.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total catchments | 242 (filtered for >30% flow data availability) |
| Temporal coverage | 1980-01-01 onwards (daily) |
| Static feature dimensions | 211 (198 numerical, 13 categorical) |
| Dynamic feature dimensions | 18 |
| Missing values in static data | 3,463 (imputed with column median) |
| Total rows across all clusters | 2,875,392 |
| Columns per combined cluster file | 74 |

### Clustering-Based Data Splits

After clustering (see Section 3), per-cluster data volumes (from `output/merged_clustered/cluster_summary_new.csv`, the merged dataset actually used for training) are:

| Cluster | Gauges | Files Found | Rows | Dominant Basins |
|---------|--------|-------------|------|-----------------|
| C0 | 65 | 56 | 838,656 | Krishna, Cauvery, EFRS |
| C1 | 66 | 49 | 733,824 | Godavari, Mahanadi, Brahmani-Baitarani |
| C2 | 37 | 36 | 539,136 | WFRS, Cauvery, Krishna |
| C3 | 74 | 51 | 763,776 | Northern basins |
| **Total** | **242** | **192** | **2,875,392** | |

> Note: "Gauges" is the count of catchments assigned to each cluster; "Files Found" is the number of per-gauge dynamic CSVs successfully located and merged. Not every assigned gauge has a usable dynamic record, which is why files < gauges in some clusters and the totals differ.

### Data Preprocessing Pipeline

1. **Static data assembly**: Merged 211 attributes from CAMELS-IND attribute CSV files for all 242 gauges
2. **Dynamic data merging**: Combined daily forcing data with baseflow time series per gauge, joined on date
3. **Missing value handling**: Static features imputed with median; dynamic NaN rows filled with column mean
4. **Categorical encoding**: 11 categorical static features (e.g., `geol_class_1st`, `dom_land_cover`, `river_basin`) encoded via `OrdinalEncoder`
5. **Normalization**: `StandardScaler` applied to dynamic features; `GroupNormalizer` with softplus transformation for TFT target
6. **Temporal indexing**: Created `time_idx` for sequential ordering; extracted `month`, `day`, `year` as known categorical covariates

### Challenges

- **Missing data**: 3,463 missing static attribute values across 242 gauges
- **Heterogeneous record lengths**: Not all gauges have continuous daily records from 1980; filtered to gauges with >30% data availability
- **Extreme values**: Baseflow ranges span several orders of magnitude (e.g., top values exceed 19,000 mm/day for large basins)

---

## 3. Methodology

### Overall Approach

The project follows a **cluster-then-predict** paradigm:

1. **Dimensionality reduction** of 197 numerical static attributes using UMAP (to 40 dimensions)
2. **Unsupervised clustering** of 242 catchments into hydroclimatically homogeneous groups
3. **Cluster-specific model training** using Temporal Fusion Transformers (TFT) and LSTMs
4. **Cross-validated evaluation** with expanding-window time-series splits

### Clustering Pipeline

**Dimensionality Reduction:**
- PCA analysis showed 52 components needed for 95% variance; top 40 components explain 91.7%
- UMAP tested at 2, 5, 10, 20, 40 dimensions; 40D selected for best distance correlation (0.4675)
- UMAP configuration: `n_neighbors=15`, `min_dist=0.1`, `random_state=42`

**Clustering Methods Compared:**

| Method | Description |
|--------|-------------|
| K-Means | Centroid-based partitioning with silhouette-optimized k |
| Gaussian Mixture Model (GMM) | Probabilistic soft clustering with full covariance |
| Enhanced Growing Neural Gas (GNG) | Topology-preserving unsupervised neural network |
| HDBSCAN | Density-based hierarchical clustering (`min_cluster_size=10`, `min_samples=5`) |
| DBSCAN | Density-based clustering tested across multiple eps values (0.1--3.0) |

**Silhouette analysis** over k=2 to k=19 identified k=4 as optimal. K-Means with k=4 was selected for producing geographically coherent regions aligned with known river basin boundaries.

**Geographic Distribution of Clusters:**

| Cluster | Lat Range | Lon Range | Center | Dominant Basins |
|---------|-----------|-----------|--------|-----------------|
| C0 | 9.32 - 20.20 | 72.93 - 80.82 | (14.70, 77.05) | Krishna, Cauvery, EFRS |
| C1 | 16.79 - 23.20 | 73.65 - 86.92 | -- | Godavari, Mahanadi, Brahmani-Baitarani |
| C2 | 8.31 - 15.47 | 74.10 - 77.83 | -- | WFRS, Cauvery, Krishna |
| C3 | Northernmost gauges | -- | -- | Northern basins |

### Model Architectures

#### 3.1 LSTM Baseline (Single-Gauge)

A 3-layer stacked LSTM trained on gauge 03005 as a proof-of-concept:

```
Input (seq_len=20, features=220)
  -> LSTM(220, 256) -> Dropout(0.5) -> ReLU
  -> LSTM(256, 128) -> Dropout(0.5) -> ReLU
  -> LSTM(128, 32)  -> Dropout(0.5) -> ReLU
  -> Linear(32, 1)
```

The 220 input features comprise 27 dynamic features + 193 static features (replicated across the temporal dimension and concatenated).

#### 3.2 Temporal Fusion Transformer (TFT) -- Primary Model

The TFT is an attention-based architecture designed for multi-horizon time-series forecasting with built-in interpretability. Key components include:

- **Variable Selection Networks**: Learns which input features are most relevant at each time step
- **Gated Residual Networks (GRN)**: Provides nonlinear processing with skip connections
- **Multi-Head Attention**: Captures long-range temporal dependencies across the encoder window
- **Static Enrichment**: Integrates static catchment attributes into temporal processing

**TFT Hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 108 |
| Attention heads | 2 |
| Dropout | 0.2 |
| Hidden continuous size | 8 |
| Encoder length | 18 (also tested 14, 10) |
| Prediction length | 1 |
| Learning rate | 0.001 |
| Parameters | ~843.7k |
| Loss function | RMSE |
| Target normalization | GroupNormalizer (softplus) |

**Feature configuration in TFT** (as set in `training/cv/tft-cv-model.ipynb`):
- **Static reals**: 39 catchment attributes — the *without-hydrological-signature* feature set is the one active in the notebook (elevation, slope, `ghi_area`, `dpsbar`, precipitation climatology, soil texture/water-holding, geology, land cover, `wtd`, `total_storage`, `reservoir_index`). The 49-feature variant that additionally includes the hydrological summaries (`bfi`, `runoff_ratio`, `q_mean`, flow statistics) is present but commented out.
- **Static categoricals**: none (`static_categoricals=[]`)
- **Time-varying known reals**: 18 dynamic forcing variables
- **Time-varying known categoricals**: `month`, `day`, `year` (cast to string)
- **Time-varying unknown reals**: baseflow (target)
- **Target normalizer**: `GroupNormalizer(groups=["group"], transformation="softplus")`, with `add_relative_time_idx`, `add_target_scales`, and `add_encoder_length` enabled

#### 3.3 KNN-Guided Models

Hybrid approaches combining K-nearest-neighbor similarity search with deep learning:

- **KNN + TFT**: Identifies similar gauges via KNN, trains TFT on combined data
- **KNN + Combined LSTM**: KNN-selected gauge data fed into shared LSTM
- **KNN + Front LSTM**: KNN-based pre-filtering with LSTM prediction head

### Key Design Decisions

1. **UMAP over PCA**: PCA requires 52 components for 95% variance; UMAP at 40 dimensions achieves superior distance correlation (0.4675) while preserving nonlinear manifold structure
2. **K-Means over alternatives**: K-Means with k=4 chosen for interpretable, geographically coherent clusters with zero noise assignment
3. **Per-cluster TFT models**: Training separate models per cluster reduces intra-cluster heterogeneity and improves model capacity allocation
4. **Expanding-window CV**: Respects temporal ordering; avoids future data leakage inherent in random splits
5. **Feature ablation (w/o hydro)**: Tested TFT without hydrological static features (bfi, runoff_ratio, q_mean, etc.) to assess dependence on potentially circular features

---

## 4. Training Pipeline

### Data Loading

- Per-cluster combined CSV files (`cluster_{cluster}_combined_new.csv`, 74 columns each) loaded from `output/merged_clustered/`
- NaNs filled with column means (`df.fillna(df.mean(numeric_only=True))`); `cluster` and `gauge_id` columns dropped; `month`/`day`/`year` cast to string
- Data split temporally: 70% train / 15% validation / 15% test (the active configuration). A 5-fold expanding-window CV scaffold also exists but is commented out

### Transformations

- `StandardScaler` on all dynamic features (LSTM pipeline)
- `GroupNormalizer` with softplus transformation on baseflow target (TFT pipeline)
- Categorical features (`month`, `day`, `year`) provided as known future covariates

### Loss Functions

| Loss | Formula | Used By |
|------|---------|---------|
| **LogRMSE** | `sqrt(mean((log(clamp(pred)) - log(clamp(target)))^2))` | LSTM baseline |
| **RMSE** | `sqrt(mean((pred - target)^2))` | TFT primary model |

LogRMSE was designed specifically for this project to handle the orders-of-magnitude variation in baseflow values; predictions and targets are clamped to a minimum (`eps=1e-8` in the TFT-notebook implementation) before log transformation. A custom `LogRMSE` `Metric` subclass is defined in the TFT notebook but the model is trained with plain `RMSE()`; LogRMSE serves as the LSTM-baseline loss. **KGE is computed only as a post-hoc diagnostic metric (via `kge_loss`), not as a training objective.**

### Training Configuration

**LSTM (Baseline):**

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 1000 |
| Loss function | LogRMSE |
| Sequence length | 20 |
| Device | CUDA GPU |

**TFT (Primary):**

| Setting | Value |
|---------|-------|
| Framework | pytorch_forecasting + PyTorch Lightning (`lightning.pytorch`) |
| Optimizer | Adam (per saved model card) |
| Learning rate | 0.001 |
| LR scheduling | ReduceOnPlateau (`reduce_on_plateau_patience=4`) |
| Gradient clipping | `gradient_clip_val=0.1` |
| Early stopping | Monitored on `val_loss` (patience 10, `min_delta=1e-4`) |
| Max epochs | 20 (model card) |
| Batch size | 1024 (train); validation/test/inference at `batch_size × 10` |
| Device | CUDA GPU (Kaggle) |
| Seed | 42 (`pl.seed_everything(42)`) |

### Data-Splitting Strategy

> **Correction vs. earlier draft:** The training notebook (`tft-cv-model.ipynb`) runs a **single temporal split**, not an automated multi-fold loop. The active `TimeSeriesDataSet` setup uses a chronological **70% / 15% / 15%** train / validation / test split (`train_cutoff = 0.7·N`, `val_cutoff = 0.85·N`). The best checkpoint (by validation loss) is then used to predict over the full cluster series, and final RMSE/MAE/R²/KGE are computed on that.

An expanding-window 5-fold cross-validation scaffold is also present in the notebook (fold setup in the CV-configuration cell), designed as follows:

| Fold | Training Window | Validation Window |
|------|----------------|-------------------|
| 1 | 0 -- 50% of data | 50% -- 65% |
| 2 | 0 -- 57% | 57% -- 72% |
| 3 | 0 -- 64% | 64% -- 79% |
| 4 | 0 -- 71% | 71% -- 86% |
| 5 | 0 -- 78% | 78% -- 93% |

The fold split logic is `train_end = 0.5·N + (fold/5)·(0.35·N)`, `val_end = train_end + 0.15·N`, so training data grows each fold from a 50% minimum and validation windows are 15% slices. **However, the fold-training loop itself is commented out in the current notebook**, so the reported per-cluster metrics come from the single-split best checkpoint rather than an averaged 5-fold run. The model artifacts are nonetheless named `…-cv5fold-seq18-…` reflecting the intended CV protocol.

### Experiment Tracking

- **Platform**: Weights & Biases (WandB)
- **Projects**: `spark'25-lstm-testing` (LSTM), `spark'25-lstm-random` (TFT)
- **TFT run naming**: `c-{cluster}-tft-cv-{run_num}-seq-18-w/o-hydro`
- **Logged**: `train_loss`, `val_loss` per epoch; final test metrics (`test_rmse`, `test_mae`, `test_r2`, `test_kge`); interactive prediction plots; predicted-vs-actual scatter; attention/variable-importance interpretation figures

### Hardware

- Training executed on **Kaggle GPU instances** (CUDA-enabled)
- Local development and preprocessing on Windows 11 with Anaconda/Python environment

---

## 5. Experiments

### Experiment 1: LSTM Baseline on Gauge 03005

| Property | Detail |
|----------|--------|
| **Hypothesis** | A single-gauge LSTM with concatenated static features can learn baseflow dynamics |
| **Configuration** | 3-layer LSTM, 220 features, seq_len=20, LogRMSE loss |
| **Training data** | 14,555 daily records (80/10/10 split) -> 11,644 train / 1,456 val / 1,389 test sequences |

### Experiment 2: Clustering Analysis

| Property | Detail |
|----------|--------|
| **Hypothesis** | Indian catchments can be meaningfully grouped by static attributes for regionalized modeling |
| **Methods compared** | K-Means, Enhanced GNG, GMM, HDBSCAN, DBSCAN |
| **Dimensionality reduction** | UMAP: 197 features -> 40 dimensions (distance correlation: 0.4675) |
| **Optimal clusters** | k=4 via silhouette analysis |

### Experiment 3: TFT with 5-Fold Cross-Validation (Per Cluster)

| Property | Detail |
|----------|--------|
| **Hypothesis** | TFT with expanding-window CV provides robust baseflow predictions within a hydroclimatic cluster |
| **Configuration** | hidden_size=108, attention_heads=2, dropout=0.2, hidden_continuous=8, encoder_len=18, pred_len=1, loss=RMSE, batch=1024 |
| **Clusters trained** | C0 (838,656 rows), C1 (733,824 rows), C2 (539,136 rows), C3 (763,776 rows) |
| **Split** | Single chronological 70/15/15 split; best-by-`val_loss` checkpoint used for full-series prediction (5-fold CV scaffold present but commented out) |

### Experiment 4: TFT without Hydrological Features (Ablation)

| Property | Detail |
|----------|--------|
| **Hypothesis** | Removing hydrological static features (bfi, runoff_ratio, q_mean, etc.) tests whether the model relies on potentially circular information |
| **Removed features** | `freq_q_low`, `freq_q_high`, `mean_anum_flow`, `bfi`, `runoff_ratio`, `q_mean`, `mean_swmn_flow`, `mean_atmn_flow`, `mean_wint_flow`, `mean_sumr_flow` |
| **Remaining static features** | 39 (vs. 49 with hydro) |
| **Tested configurations** | seq_len=18, seq_len=14, seq_len=10 |
| **Note** | The 39-feature *without-hydro* set is the configuration used for the final exported models (run names tagged `w/o-hydro`); the 49-feature *with-hydro* list is retained in the notebook (commented out) as the comparison baseline |

### Experiment 5: KNN + TFT Combined Model

| Property | Detail |
|----------|--------|
| **Hypothesis** | Nearest-neighbor pre-selection of similar gauges improves prediction |
| **Configuration** | KNN-based gauge selection + TFT training |

### Experiment 6: KNN + LSTM Variants

| Property | Detail |
|----------|--------|
| **Hypothesis** | KNN-guided LSTM models can leverage inter-gauge similarity for improved predictions |
| **Variants** | KNN + Combined LSTM (shared LSTM across KNN-selected gauges), KNN + Front LSTM (KNN pre-filtering with LSTM head) |

---

## 6. Evaluation Metrics

### Metrics Used

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **RMSE** | sqrt(mean((y - y_hat)^2)) | Primary loss; penalizes large errors heavily -- important for flood/baseflow extremes |
| **MAE** | mean(\|y - y_hat\|) | Robust to outliers; interpretable in original units (mm/day) |
| **R^2** | 1 - SS_res/SS_tot | Proportion of variance explained; benchmark against mean prediction |
| **KGE** | 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2) | Decomposes into correlation (r), variability ratio (alpha), bias ratio (beta); standard in hydrology |
| **NSE** | 1 - SS_res/SS_tot | Equivalent to R^2 for time series; widely used in hydrology |
| **LogRMSE** | sqrt(mean((log(y) - log(y_hat))^2)) | Handles orders-of-magnitude variation in baseflow; used for LSTM baseline |

### Metric Selection Justification

- **KGE** is the preferred metric in modern hydrology (Gupta et al., 2009) as it addresses known deficiencies of NSE (sensitivity to bias and variability)
- **RMSE** used as the training loss for TFT due to direct gradient properties
- **LogRMSE** used for LSTM to handle the highly skewed baseflow distribution

---

## 7. Results & Analysis

### Per-Cluster TFT Performance (seq_len=18, w/o hydrological features)

The final per-cluster TFT models (encoder length 18, RMSE loss, without-hydro static set) were exported to Hugging Face; the full-series RMSE is encoded in each artifact name and recorded in the prediction script (`DEFAULT_MODELS`):

| Cluster | Model Artifact | Final RMSE (mm/day) | Rows |
|---------|---------------|--------------------:|-----:|
| C0 | `tft-baseflow-c0-cv5fold-seq18-rmse185.0666` | 185.07 | 838,656 |
| C1 | `tft-baseflow-c1-cv5fold-seq18-rmse665.7481` | 665.75 | 733,824 |
| C2 | `tft-baseflow-c2-cv5fold-seq18-rmse317.5040` | 317.50 | 539,136 |
| C3 | `tft-baseflow-c3-cv5fold-seq18-rmse90.7000` | 90.70 | 763,776 |

**Observations:**

- **C3 is the best-performing cluster** (RMSE 90.70), and **C1 the worst** (RMSE 665.75). RMSE here is in the target's native units and is dominated by the large-magnitude baseflow values in each cluster (recall the largest baseflow values exceed several thousand mm/day), so absolute RMSE is not directly comparable across clusters with different flow-magnitude distributions.
- The wide spread (90 → 666) reflects intra-cluster heterogeneity in flow scale rather than purely model quality; per-cluster R²/KGE (logged to WandB as `test_r2`/`test_kge`) are the more comparable cross-cluster indicators and should be filled in from the run dashboard.

> RMSE/MAE/R²/KGE for the held-out 15% test split and the full-series evaluation are logged per run to WandB (`test_rmse`, `test_mae`, `test_r2`, `test_kge`) and saved in each model's `model_card.json` on Hugging Face. Populate the table below from those sources to complete the analysis.

| Cluster | Test RMSE | Test MAE | Test R² | Test KGE |
|---------|-----------|----------|---------|----------|
| C0 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| C1 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| C2 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| C3 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |

---

## 8. Ablation Studies

### Ablation 1: Hydrological Feature Removal

- **What was changed**: Removed 10 hydrological static features (`bfi`, `runoff_ratio`, `q_mean`, `freq_q_low`, `freq_q_high`, `mean_anum_flow`, `mean_swmn_flow`, `mean_atmn_flow`, `mean_wint_flow`, `mean_sumr_flow`)
- **Why**: These features are derived from the streamflow signal being predicted, introducing potential circularity. Testing without them assesses whether the model can predict baseflow from purely exogenous catchment attributes.
- **Configurations tested**: With 49 static features (full) vs. 39 static features (w/o hydro)

### Ablation 2: Encoder Sequence Length

- **What was changed**: Encoder context window varied across 18, 14, and 10 time steps
- **Why**: Longer sequences capture antecedent moisture conditions and seasonal patterns but increase computational cost and may introduce noise
- **Configurations tested**: seq_len = {18, 14, 10}, with and without hydrological features

### Ablation 3: Clustering Method

- **What was changed**: Compared K-Means, GMM, GNG, HDBSCAN, and DBSCAN for catchment grouping
- **Why**: Different clustering algorithms make different assumptions about cluster geometry (convex vs. arbitrary shape, fixed vs. variable number of clusters)
- **Outcome**: K-Means selected for producing geographically coherent 4-cluster partition

*Quantitative ablation results to be populated.*

---

## 9. Limitations

### Model Limitations

- **Single-step prediction**: The model predicts only 1 day ahead; multi-horizon forecasting was not explored
- **No ensemble methods**: Each cluster uses a single best-fold model; ensembling across folds could improve robustness
- **Fixed cluster assignment**: Catchments near cluster boundaries may be misclassified; soft clustering or overlapping approaches were not used

### Dataset Limitations

- **Baseflow separation method**: Baseflow is not directly observed but estimated via the Boughton method; errors in separation propagate to model training
- **Temporal coverage gaps**: Not all 242 gauges have continuous records; missing periods are filled rather than masked
- **Static feature circularity**: Including `bfi`, `runoff_ratio`, and `q_mean` as static features is potentially circular since they are derived from the streamflow signal being predicted
- **Single dataset**: No external validation on independent catchments outside CAMELS-IND

### Generalization Issues

- Models are trained and evaluated within the CAMELS-IND domain; transferability to other regions (e.g., CAMELS-US, CAMELS-GB) is unknown
- Cluster assignments are fixed; catchments near cluster boundaries may be misclassified

---

## 10. Future Work

1. **Multi-horizon prediction**: Extend prediction length beyond 1 day to support operational forecasting (7-day, 30-day horizons)
2. **Sub-cluster refinement**: For underperforming clusters, introduce hierarchical sub-clustering or gauge-weighted loss functions
3. **Ensemble across CV folds**: Average predictions from all 5 folds to reduce variance
4. **Transfer learning**: Pre-train on all clusters, fine-tune on target cluster to leverage shared hydrological knowledge
5. **Alternative architectures**: Evaluate Transformers (PatchTST, iTransformer), Neural ODE-based models, and graph neural networks that encode spatial catchment connectivity
6. **Remove circular features**: Train production models exclusively without hydrological summary statistics
7. **Uncertainty quantification**: Leverage TFT's quantile regression capability (currently unused) for probabilistic predictions
8. **Scaling**: Extend to all CAMELS-IND catchments (not just 242 with >30% availability) using data imputation or semi-supervised approaches
9. **Real-time deployment**: Package the best models as a prediction service for operational hydrology agencies (CWC)

---

## 11. Reproducibility

### Steps to Reproduce

1. **Data preparation**:
   - Obtain CAMELS-IND dataset (static attributes + dynamic forcings + streamflow)
   - Run `data_preprocess.ipynb` to merge static and dynamic features per gauge
   - Output: `gauge_data/static_all_gauges.csv`, per-gauge dynamic CSVs

2. **Clustering**:
   - Run `clustering.ipynb`
   - Produces: `output/clustered_gauges.csv` (242 gauges with cluster labels)
   - Produces: `output/merged_clustered/cluster_*_combined_new.csv`

3. **Model training**:
   - Run `training/cv/tft-cv-model.ipynb` on Kaggle (or any CUDA-enabled environment)
   - Set `cluster = {0, 1, 2, 3}` (and `run_num`) at the top of the notebook to train per cluster
   - The notebook loads `cluster_{cluster}_combined_new.csv`, fills NaNs with column means, builds the `TimeSeriesDataSet` on a single 70/15/15 chronological split, trains the TFT, and uploads the best checkpoint to Hugging Face
   - (Optional) Uncomment the CV-loop cell to run the 5-fold expanding-window protocol instead of the single split

4. **Inference / Evaluation**:
   - Use `predict.py` (repo root; the standalone version of the notebook's prediction system). It downloads the per-cluster model + `model_card.json` from Hugging Face (`DEFAULT_MODELS`), rebuilds the `TimeSeriesDataSet`, runs predictions, prints RMSE/MAE/R²/KGE, and writes a `y_true`/`y_pred` CSV
   - Per-cluster predictions saved as `seq18-predictions_c{cluster}_complete.csv`
   - Attention / variable-importance interpretations logged to WandB and saved under `plots_results/`

### Dependencies

```
pandas
numpy
scikit-learn
torch (PyTorch)
pytorch-forecasting
pytorch-lightning (lightning)
umap-learn
hdbscan
matplotlib
seaborn
plotly
folium
geopandas
shapely
wandb
huggingface_hub
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| TFT hidden size | 108 |
| TFT attention heads | 2 |
| TFT dropout | 0.2 |
| TFT hidden continuous size | 8 |
| Encoder length | 18 |
| Prediction length | 1 |
| Batch size | 1024 |
| Train/val/test split | 0.70 / 0.15 / 0.15 (chronological) |
| CV folds (scaffold) | 5 (expanding window, min train fraction 0.5) — currently disabled |
| Max epochs | 20 |
| Static features used | 39 (without-hydro set) |
| UMAP components | 40 |
| K-Means clusters | 4 |

### Model Artifacts

Trained per-cluster TFT models uploaded to Hugging Face under user `akshatshaw`, named `tft-baseflow-c{cluster}-cv5fold-seq18-rmse{RMSE}-timestamp{ts}`:

- C0 — `tft-baseflow-c0-cv5fold-seq18-rmse185.0666-timestamp20260328_113728`
- C1 — `tft-baseflow-c1-cv5fold-seq18-rmse665.7481-timestamp20260329_111459`
- C2 — `tft-baseflow-c2-cv5fold-seq18-rmse317.5040-timestamp20260330_074905`
- C3 — `tft-baseflow-c3-cv5fold-seq18-rmse90.7000-timestamp20260331_073611`

Each repo contains `model.pth` (state dict) and `model_card.json` (architecture, feature lists, metrics).

### Experiment Tracking

All runs logged to WandB:
- Project: `spark'25-lstm-testing` (LSTM experiments)
- Project: `spark'25-lstm-random` (TFT experiments)
- User: `akshatshaw47-iit-roorkee`
