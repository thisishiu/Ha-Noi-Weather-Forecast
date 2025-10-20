import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from train_tft import TFTLight, _make_geo_table  # type: ignore
from utils import normalize_district


def _resolve_model_dir(model_dir: Optional[str]) -> Path:
    base_dir = Path(__file__).resolve().parent
    if model_dir:
        p = Path(model_dir)
    else:
        env_dir = os.environ.get("MODEL_DIR", "").strip()
        p = Path(env_dir) if env_dir else (base_dir / "model")
    return p.resolve()


def evaluate_all(model_dir: Optional[str] = None, overall_only: bool = False):
    model_root = _resolve_model_dir(model_dir)
    global_path = model_root / "global_eval.csv"
    global_overall_path = model_root / "global_eval_overall.csv"

    if global_path.exists():
        try:
            df = pd.read_csv(global_path)
        except Exception:
            print("Global evaluation file is empty or unreadable.")
            return
        if df.empty:
            print("Global evaluation file is empty.")
            return
        model_type = None
        cfg_path = model_root / "global_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                model_type = str(cfg.get("model_type", "")).upper() or None
            except Exception:
                model_type = None
        if not overall_only:
            header = "Evaluation results (Global model)" if not model_type else f"Evaluation results (Global {model_type})"
            print(header + ":")

        if global_overall_path.exists():
            try:
                df_overall = pd.read_csv(global_overall_path)
                if not df_overall.empty and {"scope", "MAE", "RMSE"}.issubset(df_overall.columns):
                    df_overall.to_csv(model_root / "overall_metrics.csv", index=False)
                    if overall_only:
                        # Print compact overall metrics only
                        for _, row in df_overall.iterrows():
                            scope = row.get("scope", "?")
                            mae_o = row.get("MAE", float("nan"))
                            rmse_o = row.get("RMSE", float("nan"))
                            r2_o = row.get("R2", float("nan"))
                            print(f"{scope}: MAE={mae_o:.4f}, RMSE={rmse_o:.4f}, R2={r2_o:.4f}")
                    else:
                        print("Overall metrics:")
                        for _, row in df_overall.iterrows():
                            scope = row.get("scope", "?")
                            mae_o = row.get("MAE", float("nan"))
                            rmse_o = row.get("RMSE", float("nan"))
                            r2_o = row.get("R2", float("nan"))
                            print(f"  {scope}: MAE={mae_o:.4f}, RMSE={rmse_o:.4f}, R2={r2_o:.4f}")
            except Exception:
                pass

        if not overall_only:
            has_r2 = "R2" in df.columns
            for _, row in df.iterrows():
                district = row.get("district", "?")
                mae = row.get("MAE", float("nan"))
                rmse = row.get("RMSE", float("nan"))
                if has_r2:
                    r2 = row.get("R2", float("nan"))
                    print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
                else:
                    print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            model_root.mkdir(parents=True, exist_ok=True)
            df.to_csv(model_root / "eval_results.csv", index=False)
            print(f"Evaluation summary saved -> {model_root / 'eval_results.csv'}")
        return

    print("No evaluation files found. Run training first.")


def run_evaluate(model_dir: Optional[str] = None):
    evaluate_all(model_dir)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    tmp = df.copy()
    dt = pd.to_datetime(tmp["datetime"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(int)
    dow = dt.dt.dayofweek.fillna(0).astype(int)
    month = dt.dt.month.fillna(1).astype(int)
    tmp["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    tmp["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    tmp["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    tmp["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    tmp["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    tmp["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return tmp


def _build_coords(raw_csv: Path) -> dict:
    try:
        df = pd.read_csv(raw_csv, usecols=["district", "lat", "lon"])
    except Exception:
        df = pd.read_csv(raw_csv, usecols=["district", "lat", "lon"], encoding="utf-8", errors="ignore")
    df["district"] = df["district"].astype(str).map(normalize_district)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    coords = (
        df.groupby("district", as_index=True)[["lat", "lon"]]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    return {k: (v["lat"], v["lon"]) for k, v in coords.items()}


def compute_overall(model_dir: Optional[str] = None) -> Path:
    model_root = _resolve_model_dir(model_dir)
    cfg_path = model_root / "global_config.json"
    model_path = model_root / "global_tft.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model or config in {model_root}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_root = Path(__file__).resolve().parents[1]
    test_csv = base_root / "data/splits/test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run preprocess first.")

    hdr = pd.read_csv(test_csv, nrows=0)
    parse_dates = ["datetime"] if "datetime" in hdr.columns else None
    test_df = pd.read_csv(test_csv, parse_dates=parse_dates)
    test_df = _add_time_features(test_df)

    district2idx = {k: int(v) for k, v in cfg["district2idx"].items()}
    feature_names = list(cfg.get("feature_names", []))
    target_features = list(cfg.get("target_features", [])) or feature_names
    target_idx = [feature_names.index(f) for f in target_features]
    # IMPORTANT: construct the model with ORIGINAL training output size to load checkpoint
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])
    lookback = int(cfg["lookback"])
    horizon = int(cfg.get("horizon", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = _build_coords(base_root / "data/hanoi_weather.csv")
    geo_table = _make_geo_table(district2idx, coords, fourier_K=int(cfg.get("fourier_K", 2)))
    model = TFTLight(
        num_features,
        num_districts,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        geo_table=geo_table,
        geo_emb_dim=int(cfg.get("geo_emb_dim", 8)),
        id_emb_dim=int(cfg.get("id_emb_dim", 8)),
        horizon=horizon,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    total_abs_err = 0.0
    total_sq_err = 0.0
    total_count = 0
    sum_y_vec = None
    sum_y2_vec = None

    for district, g in test_df.groupby("district", sort=False):
        key = str(district)
        if key not in district2idx:
            alt = "_".join([w.capitalize() for w in key.split("_")])
            if alt in district2idx:
                key = alt
            else:
                continue
        d_idx = district2idx[key]
        feature_names = list(cfg.get("feature_names", []))
        target_features = list(cfg.get("target_features", [])) or feature_names
        target_idx = [feature_names.index(f) for f in target_features]
        if feature_names:
            feat_df = g[feature_names]
        else:
            feat_df = g.select_dtypes(include=[float, int, np.number])
        min_len = lookback + horizon
        if feat_df.empty or len(feat_df) < min_len:
            continue
        values = feat_df.values.astype(np.float32)
        X_list, y_list = [], []
        for start in range(0, len(values) - min_len + 1):
            X_list.append(values[start : start + lookback])
            y_list.append(values[start + lookback : start + lookback + horizon])
        X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
        y = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
        d = torch.full((X.shape[0],), d_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(X, d)
        pred = pred[:, :, target_idx]
        y = y[:, :, target_idx]

        err = pred - y
        total_abs_err += torch.sum(torch.abs(err)).item()
        total_sq_err += torch.sum(err * err).item()
        total_count += y.shape[0] * y.shape[1]
        y_sum = torch.sum(y, dim=(0, 1)).detach().cpu().numpy()
        y_sum2 = torch.sum(y * y, dim=(0, 1)).detach().cpu().numpy()
        if sum_y_vec is None:
            sum_y_vec = y_sum
            sum_y2_vec = y_sum2
        else:
            sum_y_vec += y_sum
            sum_y2_vec += y_sum2

    model_root.mkdir(parents=True, exist_ok=True)
    denom = max(1, total_count * num_features)
    micro_mae = total_abs_err / denom
    micro_rmse = float(np.sqrt(total_sq_err / denom))
    if total_count > 0 and sum_y_vec is not None and sum_y2_vec is not None:
        mean_vec = sum_y_vec / float(total_count)
        ss_tot_total = float(np.sum(sum_y2_vec - float(total_count) * (mean_vec ** 2)))
        if ss_tot_total == 0.0:
            micro_r2 = 1.0 if total_sq_err == 0.0 else 0.0
        else:
            micro_r2 = 1.0 - (total_sq_err / ss_tot_total)
    else:
        micro_r2 = float("nan")

    overall_df = pd.DataFrame([
        {"scope": "micro", "MAE": micro_mae, "RMSE": micro_rmse, "R2": micro_r2}
    ])
    out_path = model_root / "overall_metrics.csv"
    overall_df.to_csv(out_path, index=False)
    # Backward compatibility
    overall_df.to_csv(model_root / "global_eval_overall.csv", index=False)
    print(f"Overall metrics saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TFT model results")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model directory")
    parser.add_argument("--overall-only", action="store_true", help="Print/save only overall metrics")
    parser.add_argument("--recompute", action="store_true", help="Recompute overall metrics from model and test set")
    args = parser.parse_args()
    if args.recompute:
        compute_overall(args.model_dir)
    else:
        evaluate_all(args.model_dir, overall_only=args.overall_only)
