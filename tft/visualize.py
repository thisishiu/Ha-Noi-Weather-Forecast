import os
import json
from math import ceil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils import normalize_district
from train_tft import TFTLight, _make_geo_table  # type: ignore


def plot_global_loss(save_path: str | Path | None = None, show: bool = False):
    base_dir = Path(__file__).resolve().parent
    loss_path = base_dir / "model/global_loss.csv"
    if not loss_path.exists():
        raise FileNotFoundError(f"{loss_path} not found")
    df = pd.read_csv(loss_path)
    if df.empty:
        raise ValueError("global_loss.csv is empty")
    if save_path is None:
        save_path = base_dir / "model/figures/global_loss.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Global TFT-Light Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {save_path}")


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

def _load_cfg_and_model() -> Tuple[dict, TFTLight, torch.device]:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    cfg_path = model_dir / "global_config.json"
    model_path = model_dir / "global_tft.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing model or config. Train TFT first.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])
    base_root = Path(__file__).resolve().parents[1]
    coords_csv = str(base_root / "data/hanoi_weather.csv")
    district2idx = {k: int(v) for k, v in cfg["district2idx"].items()}
    geo_table = _make_geo_table(district2idx, _build_coords(coords_csv), fourier_K=int(cfg.get("fourier_K", 2)))

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
        horizon=int(cfg.get("horizon", 6)),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return cfg, model, device


def _build_coords(raw_csv: str) -> dict:
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


def _predict_series_for_district(
    cfg: dict,
    model: TFTLight,
    device: torch.device,
    test_df: pd.DataFrame,
    district: str,
) -> Tuple[pd.Series, np.ndarray, np.ndarray, List[str]]:
    lookback = int(cfg["lookback"])
    horizon = int(cfg.get("horizon", 1))
    feature_names: List[str] = list(cfg["feature_names"]) if "feature_names" in cfg else [
        c for c in test_df.select_dtypes(include=[float, int, np.number]).columns
    ]
    dnorm = normalize_district(district)

    g = test_df[test_df["district"].astype(str) == dnorm].copy()
    min_len = lookback + horizon
    if g.empty or len(g) < min_len:
        raise ValueError(f"Not enough data for district {district} to plot.")
    g = g.sort_values("datetime").reset_index(drop=True)
    feat_df = g[feature_names]
    values = feat_df.values.astype(np.float32)

    X_list, y_list = [], []
    for start in range(0, len(values) - min_len + 1):
        X_list.append(values[start : start + lookback])
        y_list.append(values[start + lookback : start + lookback + horizon])
    X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    d2i = {k: int(v) for k, v in cfg["district2idx"].items()}
    key = dnorm
    alt = "_".join([w.capitalize() for w in key.split("_")])
    if alt in d2i:
        map_key = alt
    elif key in d2i:
        map_key = key
    else:
        raise KeyError(f"District {district} not found in model mapping.")
    d_idx = torch.full((X.shape[0],), d2i[map_key], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, d_idx).detach().cpu().numpy()  # [N,H,F]
    y_true = np.stack(y_list)
    pred_1 = pred[:, 0, :]
    y_true_1 = y_true[:, 0, :]
    times = g["datetime"].iloc[lookback : lookback + pred_1.shape[0]].reset_index(drop=True)
    return times, y_true_1, pred_1, feature_names


def plot_pred_vs_actual_all(district: str | None = None, show: bool = False):
    base_dir = Path(__file__).resolve().parent
    base_root = base_dir.parents[0]
    test_csv = base_root / "data/splits/test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run preprocess first.")
    hdr = pd.read_csv(test_csv, nrows=0)
    parse_dates = ["datetime"] if "datetime" in hdr.columns else None
    test_df = pd.read_csv(test_csv, parse_dates=parse_dates)
    test_df = _add_time_features(test_df)

    cfg, model, device = _load_cfg_and_model()

    if district:
        targets = [district]
    else:
        env_d = os.environ.get("PLOT_DISTRICT", "").strip()
        targets = [env_d] if env_d else (list(cfg.get("district2idx", {}).keys()) or list(sorted(test_df["district"].astype(str).unique())))

    out_dir = base_dir / "model/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in targets:
        try:
            times, y_true, y_pred, feature_names = _predict_series_for_district(cfg, model, device, test_df, tgt)
        except Exception as e:
            print(f"Skip {tgt}: {e}")
            continue

        n_feat = len(feature_names)
        cols = 2
        rows = ceil(n_feat / cols)
        plt.figure(figsize=(cols * 6, rows * 2.5))
        for i, fname in enumerate(feature_names):
            ax = plt.subplot(rows, cols, i + 1)
            ax.plot(times, y_true[:, i], label="Actual", linewidth=1.2)
            ax.plot(times, y_pred[:, i], label="Pred", linewidth=1.0)
            ax.set_title(fname)
            ax.tick_params(axis="x", labelrotation=30)
        plt.suptitle(f"TFT-Light Pred vs Actual - {normalize_district(tgt)}")
        plt.legend(loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f"global_pred_vs_actual_all_{normalize_district(tgt)}.png"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()
        print(f"Saved: {out_path}")


def plot_horizon_detail(district: str, samples: int = 200, feature: str | None = None, show: bool = False):
    base_dir = Path(__file__).resolve().parent
    base_root = base_dir.parents[0]
    test_csv = base_root / "data/splits/test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run preprocess first.")

    hdr = pd.read_csv(test_csv, nrows=0)
    parse_dates = ["datetime"] if "datetime" in hdr.columns else None
    test_df = pd.read_csv(test_csv, parse_dates=parse_dates)
    test_df = _add_time_features(test_df)

    cfg, model, device = _load_cfg_and_model()

    # Prepare feature names and district mapping
    feature_names: List[str] = list(cfg.get("feature_names", []))
    d2i = {k: int(v) for k, v in cfg["district2idx"].items()}
    dnorm = normalize_district(district)
    key = dnorm
    alt = "_".join([w.capitalize() for w in key.split("_")])
    if alt in d2i:
        map_key = alt
    elif key in d2i:
        map_key = key
    else:
        raise KeyError(f"District {district} not found in model mapping.")

    # Slice district
    g = test_df[test_df["district"].astype(str) == dnorm].copy()
    lookback = int(cfg["lookback"])
    horizon = int(cfg.get("horizon", 1))
    min_len = lookback + horizon
    if g.empty or len(g) < min_len:
        raise ValueError(f"Not enough data for district {district} to plot.")
    g = g.sort_values("datetime").reset_index(drop=True)
    feat_df = g[feature_names]
    values = feat_df.values.astype(np.float32)

    # Build windows
    X_list, y_list = [], []
    for start in range(0, len(values) - min_len + 1):
        X_list.append(values[start : start + lookback])
        y_list.append(values[start + lookback : start + lookback + horizon])
    X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    d_idx = torch.full((X.shape[0],), d2i[map_key], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, d_idx).detach().cpu().numpy()  # [N,H,F]
    y_true = np.stack(y_list)  # [N,H,F]

    # Choose feature
    if feature and feature in feature_names:
        fidx = feature_names.index(feature)
        feat_name = feature
    else:
        if "temperature_2m" in feature_names:
            fidx = feature_names.index("temperature_2m")
            feat_name = "temperature_2m"
        else:
            fidx = 0
            feat_name = feature_names[0] if feature_names else "feature_0"

    N = min(int(samples), pred.shape[0])
    pred = pred[:N, :, fidx]
    y_true = y_true[:N, :, fidx]

    times1 = g["datetime"].iloc[lookback : lookback + N].reset_index(drop=True)

    cols = 1
    rows = horizon
    plt.figure(figsize=(10, 2.2 * rows))
    for h in range(horizon):
        ax = plt.subplot(rows, cols, h + 1)
        t_h = times1 + pd.to_timedelta(h + 1, unit="h") if "datetime" in g.columns else np.arange(N)
        ax.plot(t_h, y_true[:, h], label="Actual", linewidth=1.2)
        ax.plot(t_h, pred[:, h], label="Pred", linewidth=1.0)
        ax.set_title(f"{feat_name} - horizon t+{h+1}h")
        if h == rows - 1:
            ax.tick_params(axis="x", labelrotation=30)
        else:
            ax.tick_params(axis="x", labelbottom=False)
    plt.suptitle(f"TFT-Light Horizon Detail - {dnorm}")
    plt.legend(loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = base_dir / f"model/figures/horizon_detail_{dnorm}_{feat_name}.png"
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")


def run_visualize():
    try:
        plot_global_loss()
    except Exception as e:
        print(f"Skip global loss plot: {e}")
    try:
        env_d = os.environ.get("PLOT_DISTRICT", "").strip()
        if env_d:
            samples = int(os.environ.get("PLOT_SAMPLES", "200") or 200)
            feats_csv = os.environ.get("PLOT_FEATURES", "").strip()
            single_feat = os.environ.get("PLOT_FEATURE", "").strip()
            if feats_csv:
                feats = [f.strip() for f in feats_csv.split(",") if f.strip()]
            elif single_feat:
                feats = [single_feat]
            else:
                # Choose a few common features if not specified
                try:
                    cfg, _m, _d = _load_cfg_and_model()
                    all_feats = list(cfg.get("feature_names", []))
                except Exception:
                    all_feats = []
                preferred = [
                    "temperature_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "relative_humidity_2m",
                ]
                feats = [f for f in preferred if f in all_feats] or (all_feats[:3] if all_feats else [])
            max_feats = int(os.environ.get("PLOT_MAX_FEATURES", "6") or 6)
            for f in feats[:max_feats]:
                plot_horizon_detail(env_d, samples=samples, feature=f or None)
        else:
            plot_pred_vs_actual_all(None)
    except Exception as e:
        print(f"Skip pred-vs-actual plot: {e}")


if __name__ == "__main__":
    run_visualize()
