Weather Forecast (District-level) — TFT Pipeline

This repository implements a TFT-Light (Transformer-based) multi-horizon forecaster for district-level weather. It includes preprocessing, training, evaluation, visualization, and ONNX export for R (Shiny/Plumber).

Sections
- Overview
- Structure
- Data & Preprocessing
- Quickstart (Train)
- Artifacts
- Visualization
- Evaluate (Test)
- ONNX Export
- Use ONNX in R
- Defaults & Tips

Overview
- Model: TFT-Light (Transformer encoder + district/geo embeddings + horizon positional embedding)
- Inputs: lookback window per district (default lookback=48 hours)
- Outputs: multi-horizon for all features (default horizon=6 hours)
- Splits: data/splits/{train,dev,test}.csv
- Artifacts: tft/model/

Structure
- tft/preprocess.py — build data/splits, add time features, scale
- tft/train_tft.py — train, save model/config/metrics, evaluate
- tft/visualize.py — plot loss, per-district Pred vs Actual, horizon-detail
- tft/run_eval.py — evaluate saved model on test
- tft/export_onnx.py — export ONNX
- tft/model/ — global_tft.pt, global_config.json, global_loss.csv, global_eval*.csv, figures/

Data & Preprocessing
- Raw: data/hanoi_weather.csv (required: datetime, district, lat, lon)
- Core features (examples): temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature, surface_pressure, precipitation/rain, cloud_cover, wind_speed_10m
- Time features from datetime: hour_sin/cos, dow_sin/cos, month_sin/cos
- Scaling: MinMax fitted on train, applied to dev/test
- Optional: LOG1P_RAIN=1 applies log1p to precipitation/rain before scaling

Quickstart (Train)
1) Install deps: `pip install -r requirements.txt`
2) Run training (PowerShell examples):
```
# Optional toggles
# $env:FORCE_PREPROCESS="1"
# $env:LOG1P_RAIN="1"
# $env:HUBER_BETA="0.25"
python tft/main.py
```
This builds splits (if needed), trains TFT-Light, evaluates on test, and writes plots.

Artifacts
- tft/model/global_tft.pt — weights
- tft/model/global_config.json — lookback, horizon, feature_names, district2idx, model dims
- tft/model/global_loss.csv — loss per epoch
- tft/model/global_eval.csv — per-district metrics
- tft/model/global_eval_overall.csv — micro/macro aggregates
- tft/model/figures/global_loss.png — learning curve
- tft/model/figures/global_pred_vs_actual_all_<district>.png — per-district 1-step overlay
- tft/model/figures/horizon_detail_<district>_<feature>.png — t+1..t+6 subplots for one feature

Visualization
- All plots: `python tft/visualize.py`
- One district (horizon-detail):
```
$env:PLOT_DISTRICT="thanh_tri"
$env:PLOT_SAMPLES="200"
$env:PLOT_FEATURES="temperature_2m,precipitation,wind_speed_10m"  # optional
python tft/visualize.py
```

Evaluate (Test)
```
python tft/run_eval.py
```
Writes tft/model/global_eval.csv and tft/model/global_eval_overall.csv.

ONNX Export
```
python tft/export_onnx.py
```
- ONNX: tft/model/global_tft.onnx
- Config: tft/model/global_config.json
- Inputs: X [batch, time, num_features] float32 (time >= lookback), district_idx [batch] int64
- Output: y_pred [batch, horizon, num_features] float32

Use ONNX in R (Shiny/Plumber)
- Requirements: onnxruntime, jsonlite, data.table
- Steps: load config + ONNX; compute time features; scale by train stats in feature_names order; build X [1, lookback, F] and district_idx [1]; run ONNX to get y_pred [1, H, F].
- Minimal R example is in comments within this README’s earlier instructions (mirror Python logic).

Defaults & Tips
- Defaults: lookback=48, horizon=6, d_model=192, nhead=4, num_layers=3, dropout=0.1
- Scheduler: OneCycleLR (batch-level step)
- Loss: Huber with horizon weighting (gamma=0.5)
- Env toggles: FORCE_PREPROCESS, LOG1P_RAIN, HUBER_BETA; viz toggles: PLOT_DISTRICT, PLOT_SAMPLES, PLOT_FEATURE(S)

