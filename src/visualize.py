import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

try:
    from train_lstm import LSTMModel
except Exception:
    LSTMModel = None

def plot_loss(district):
    df = pd.read_csv(f"model/lstm/{district}_loss.csv")
    plt.figure(figsize=(6,4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve - {district}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_actual(district, lookback=24):
    if LSTMModel is None:
        raise RuntimeError("LSTMModel not available; ensure train_lstm.py exists and is importable.")
    test = pd.read_csv(f"data/splits/test/{district}.csv").select_dtypes(float).values
    X, y = [], []
    for i in range(len(test) - lookback):
        X.append(test[i:i+lookback])
        y.append(test[i+lookback])
    X, y = np.array(X), np.array(y)
    model = LSTMModel(X.shape[2])
    model.load_state_dict(torch.load(f"model/lstm/{district}_lstm.pt"))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()

    plt.figure(figsize=(10,4))
    plt.plot(y[:100, 0], label="Actual", color="black")
    plt.plot(preds[:100, 0], label="Predicted", color="orange")
    plt.title(f"Prediction vs Actual - {district}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from pathlib import Path
    data_root = Path("data/splits/train")
    for f in data_root.glob("*.csv"):
        district = f.stem
        plot_loss(district)
        plot_pred_vs_actual(district)


def run_visualize():
    """Wrapper to match main.py expectation.

    Generates global training loss plot and district-level
    prediction vs actual plots for the global model.
    Saves figures to model/figures.
    
    Behavior:
    - If env `PLOT_DISTRICT` is set: plot only that district.
    - Else: plot for ALL districts found in evaluation/config.
    """
    try:
        plot_global_loss()
    except Exception as e:
        print(f"Skip global loss plot: {e}")

    eval_path = Path("model/global_eval.csv")
    df = None
    if eval_path.exists():
        try:
            df = pd.read_csv(eval_path)
        except Exception as e:
            print(f"Cannot read global_eval.csv: {e}")

    target = os.environ.get("PLOT_DISTRICT")
    if target:
        try:
            plot_global_pred_vs_actual_all(target)
        except Exception as e:
            print(f"Skip global pred-vs-actual(all) for {target}: {e}")
        return

    # Plot for all districts
    districts = []
    if df is not None and "district" in df.columns and not df.empty:
        try:
            districts = [str(d) for d in df["district"].dropna().unique().tolist()]
        except Exception:
            districts = []
    print(f"Plotting pred-vs-actual(all) for {len(districts)} districts...")
    try:
        plot_global_pred_vs_actual_all_multi(districts if districts else None)
    except Exception as e:
        print(f"Plot-multi failed, falling back to per-district: {e}")
        for dname in districts:
            try:
                plot_global_pred_vs_actual_all(dname)
            except Exception as e2:
                print(f"Skip district {dname}: {e2}")


# ===== Global model visualization =====

def _load_global_model_and_config():
    cfg_path = Path("model/global_config.json")
    model_path = Path("model/global_lstm.pt")
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing global model or config. Run training first.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])
    from train_global import GlobalLSTM, _build_coords_from_raw, _make_geo_table  # local import
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = _build_coords_from_raw("data/hanoi_weather.csv")
    geo_table = _make_geo_table(cfg["district2idx"], coords, fourier_K=int(cfg.get("fourier_K", 2)))
    model = GlobalLSTM(
        num_features,
        num_districts,
        emb_dim=int(cfg["emb_dim"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg.get("dropout", 0.1)),
        geo_table=geo_table,
        geo_emb_dim=int(cfg.get("geo_emb_dim", 8)),
        use_id_emb=True,
        post_dropout=float(cfg.get("post_dropout", 0.0)),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg, device


def plot_global_loss(save_path: str | Path = "model/figures/global_loss.png", show: bool = False):
    loss_path = Path("model/global_loss.csv")
    if not loss_path.exists():
        raise FileNotFoundError("model/global_loss.csv not found")
    df = pd.read_csv(loss_path)
    if df.empty:
        raise ValueError("global_loss.csv is empty")
    fig_dir = Path(save_path).parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Global LSTM Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def plot_global_pred_vs_actual(district: str, save_dir: str | Path = "model/figures", points: int = 200):
    model, cfg, device = _load_global_model_and_config()
    lookback = int(cfg["lookback"])
    feature_names = cfg.get("feature_names", [])
    district2idx = cfg["district2idx"]
    if district not in district2idx:
        raise ValueError(f"District '{district}' not found in model config")

    test_csv = Path("data/splits/test.csv")
    if not test_csv.exists():
        raise FileNotFoundError("data/splits/test.csv not found")
    df = pd.read_csv(test_csv)
    g = df[df["district"] == district]
    feat_df = g.select_dtypes(include=[float, int, np.number])
    if feat_df.empty or len(feat_df) <= lookback:
        raise ValueError("Not enough test data for plotting")
    values = feat_df.values.astype(np.float32)

    # choose a feature to show: temperature_2m if available else first
    show_idx = 0
    if feature_names:
        try:
            show_idx = feature_names.index("temperature_2m")
        except ValueError:
            show_idx = 0

    X_list, y_list = [], []
    for start in range(0, len(values) - lookback):
        X_list.append(values[start : start + lookback])
        y_list.append(values[start + lookback])
    X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    d_idx = int(district2idx[district])
    d = torch.full((X.shape[0],), d_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, d).cpu().numpy()
    y = np.stack(y_list)

    # limit points for readability
    n = min(points, len(y))
    fig_dir = Path(save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(y[:n, show_idx], label="Actual", color="black")
    plt.plot(pred[:n, show_idx], label="Predicted", color="orange")
    fname = fig_dir / f"global_pred_vs_actual_{district}.png"
    plt.title(f"Global LSTM - {district} ({feature_names[show_idx] if feature_names else 'feature_0'})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_global_pred_vs_actual_all(district: str, save_dir: str | Path = "model/figures", points: int = 200, cols: int = 2):
    """Plot Pred vs Actual for all forecasted features of one district (test split).

    - Uses model/global_lstm.pt and model/global_config.json
    - Reads data/splits/test.csv (combined, normalized districts)
    - Saves a single grid figure containing all features
    """
    from math import ceil

    model, cfg, device = _load_global_model_and_config()
    lookback = int(cfg["lookback"])
    feature_names = cfg.get("feature_names", [])
    district2idx = cfg["district2idx"]
    if district not in district2idx:
        raise ValueError(f"District '{district}' not found in model config")

    test_csv = Path("data/splits/test.csv")
    if not test_csv.exists():
        raise FileNotFoundError("data/splits/test.csv not found")
    df = pd.read_csv(test_csv)
    g = df[df["district"] == district]
    feat_df = g.select_dtypes(include=[float, int, np.number])
    if feat_df.empty or len(feat_df) <= lookback:
        raise ValueError("Not enough test data for plotting")
    values = feat_df.values.astype(np.float32)

    X_list, y_list = [], []
    for start in range(0, len(values) - lookback):
        X_list.append(values[start : start + lookback])
        y_list.append(values[start + lookback])
    X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    d_idx = int(district2idx[district])
    d = torch.full((X.shape[0],), d_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, d).cpu().numpy()
    y = np.stack(y_list)

    n = min(points, len(y))
    num_features = y.shape[1]
    rows = ceil(num_features / cols)

    fig_dir = Path(save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt  # ensure local import in case headless
    plt.figure(figsize=(cols * 6, rows * 3))
    for i in range(num_features):
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(y[:n, i], label="Actual", color="black", linewidth=1.0)
        ax.plot(pred[:n, i], label="Predicted", color="orange", linewidth=1.0)
        title = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        ax.set_title(title)
        if i % cols == 0:
            ax.set_ylabel("Value")
        if i // cols == rows - 1:
            ax.set_xlabel("Time step")
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    fname = fig_dir / f"global_pred_vs_actual_all_{district}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_global_pred_vs_actual_all_multi(districts: list[str] | None = None, save_dir: str | Path = "model/figures", points: int = 200, cols: int = 2):
    """Efficiently plot Pred vs Actual for all features across multiple districts.

    Loads the model once and iterates districts. If `districts` is None,
    attempts to derive the list from `model/global_eval.csv` or from config.
    """
    from math import ceil

    model, cfg, device = _load_global_model_and_config()
    lookback = int(cfg["lookback"])
    feature_names = cfg.get("feature_names", [])
    district2idx = cfg["district2idx"]

    # Determine district list
    if districts is None:
        eval_path = Path("model/global_eval.csv")
        if eval_path.exists():
            try:
                df_eval = pd.read_csv(eval_path)
                if "district" in df_eval.columns and not df_eval.empty:
                    districts = [str(d) for d in df_eval["district"].dropna().unique().tolist()]
            except Exception:
                districts = None
    if not districts:
        districts = list(district2idx.keys())

    test_csv = Path("data/splits/test.csv")
    if not test_csv.exists():
        raise FileNotFoundError("data/splits/test.csv not found")
    df_all = pd.read_csv(test_csv)

    fig_dir = Path(save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for district in districts:
        if district not in district2idx:
            print(f"District '{district}' not in config; skip.")
            continue
        g = df_all[df_all["district"] == district]
        feat_df = g.select_dtypes(include=[float, int, np.number])
        if feat_df.empty or len(feat_df) <= lookback:
            print(f"Not enough test data for {district}; skip.")
            continue
        values = feat_df.values.astype(np.float32)

        X_list, y_list = [], []
        for start in range(0, len(values) - lookback):
            X_list.append(values[start : start + lookback])
            y_list.append(values[start + lookback])
        X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
        d_idx = int(district2idx[district])
        d = torch.full((X.shape[0],), d_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(X, d).cpu().numpy()
        y = np.stack(y_list)

        n = min(points, len(y))
        num_features = y.shape[1]
        rows = ceil(num_features / cols)

        import matplotlib.pyplot as plt  # local import for headless safety
        plt.figure(figsize=(cols * 6, rows * 3))
        for i in range(num_features):
            ax = plt.subplot(rows, cols, i + 1)
            ax.plot(y[:n, i], label="Actual", color="black", linewidth=1.0)
            ax.plot(pred[:n, i], label="Predicted", color="orange", linewidth=1.0)
            title = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            ax.set_title(title)
            if i % cols == 0:
                ax.set_ylabel("Value")
            if i // cols == rows - 1:
                ax.set_xlabel("Time step")
            if i == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()
        fname = fig_dir / f"global_pred_vs_actual_all_{district}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")
