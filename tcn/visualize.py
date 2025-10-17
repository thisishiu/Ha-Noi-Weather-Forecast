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
from train_tcn import GlobalTCN, _make_geo_table  # type: ignore


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
    plt.title("Global TCN Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def _load_cfg_and_model() -> Tuple[dict, GlobalTCN, torch.device]:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    cfg_path = model_dir / "global_config.json"
    model_path = model_dir / "global_tcn.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing model or config. Train TCN first.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model from config
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])

    # Geo features
    base_root = Path(__file__).resolve().parents[1]
    coords_csv = str(base_root / "data/hanoi_weather.csv")
    # Reuse the geo table builder from training
    # Build district2idx dict keys order is important for geo mapping
    district2idx = {k: int(v) for k, v in cfg["district2idx"].items()}
    geo_table = _make_geo_table(district2idx, _build_coords(coords_csv), fourier_K=int(cfg.get("fourier_K", 2)))

    horizon = int(cfg.get("horizon", 1))
    model = GlobalTCN(
        num_features,
        num_districts,
        emb_dim=int(cfg.get("emb_dim", 8)),
        channels=int(cfg.get("tcn_channels", 64)),
        num_blocks=int(cfg.get("tcn_num_blocks", 4)),
        kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        dropout=float(cfg.get("tcn_dropout", 0.1)),
        geo_table=geo_table,
        geo_emb_dim=int(cfg.get("geo_emb_dim", 8)),
        use_id_emb=True,
        post_dropout=float(cfg.get("post_dropout", 0.0)),
        horizon=horizon,
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
    model: GlobalTCN,
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
    # Build rolling windows with horizon
    X_list, y_list = [], []
    for start in range(0, len(values) - min_len + 1):
        X_list.append(values[start : start + lookback])
        y_list.append(values[start + lookback : start + lookback + horizon])
    X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    d2i = {k: int(v) for k, v in cfg["district2idx"].items()}
    if dnorm not in d2i:
        raise KeyError(f"District {district} not found in model mapping.")
    d_idx = torch.full((X.shape[0],), d2i[dnorm], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, d_idx).detach().cpu().numpy()  # [N,H,F]
    y_true = np.stack(y_list)  # [N,H,F]
    # For plotting 1-step ahead, take the first horizon step
    pred_1 = pred[:, 0, :]
    y_true_1 = y_true[:, 0, :]
    # Align timestamps to the first target step
    times = g["datetime"].iloc[lookback : lookback + pred_1.shape[0]].reset_index(drop=True)
    return times, y_true_1, pred_1, feature_names


def plot_pred_vs_actual_all(district: str | None = None, show: bool = False):
    base_dir = Path(__file__).resolve().parent
    base_root = base_dir.parents[1]
    # Use consistent path with training/preprocess
    test_csv = base_root / "data/splits/test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run preprocess first.")
    # Read header first to detect datetime presence
    hdr = pd.read_csv(test_csv, nrows=0)
    parse_dates = ["datetime"] if "datetime" in hdr.columns else None
    test_df = pd.read_csv(test_csv, parse_dates=parse_dates)

    cfg, model, device = _load_cfg_and_model()

    # Determine target districts
    if district:
        targets = [district]
    else:
        env_d = os.environ.get("PLOT_DISTRICT", "").strip()
        if env_d:
            targets = [env_d]
        else:
            targets = list(cfg.get("district2idx", {}).keys())
            if not targets:
                targets = list(sorted(test_df["district"].astype(str).unique()))

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
        plt.suptitle(f"TCN Pred vs Actual - {normalize_district(tgt)}")
        plt.legend(loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f"global_pred_vs_actual_all_{normalize_district(tgt)}.png"
        out_path = out_dir / fname
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
    # Try to plot Pred vs Actual for either env district or all
    try:
        env_d = os.environ.get("PLOT_DISTRICT", "").strip()
        plot_pred_vs_actual_all(env_d if env_d else None)
    except Exception as e:
        print(f"Skip pred-vs-actual plot: {e}")


if __name__ == "__main__":
    run_visualize()
