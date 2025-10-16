import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from math import pi

from utils import normalize_district


class GlobalWeatherDataset(Dataset):
    def __init__(self, df: pd.DataFrame, district2idx: Dict[str, int], lookback: int = 24):
        self.lookback = lookback
        self.district2idx = district2idx
        self.series: Dict[int, np.ndarray] = {}
        self.feature_names: List[str] = []
        self.indices: List[Tuple[int, int]] = []  # (district_idx, start_idx)

        # Expect combined DataFrame with 'district' column
        if df is None or df.empty:
            return
        feat_df = df.select_dtypes(include=[float, int, np.number])
        self.feature_names = list(feat_df.columns)

        # build per-district arrays and sliding windows (no cross-district)
        for dname, g in df.groupby("district", sort=False):
            if dname not in district2idx:
                continue
            d_idx = district2idx[dname]
            vals = g[self.feature_names].values.astype(np.float32)
            self.series[d_idx] = vals
            if len(vals) > self.lookback:
                for start in range(0, len(vals) - self.lookback):
                    self.indices.append((d_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, start = self.indices[idx]
        arr = self.series[d_idx]
        X = arr[start : start + self.lookback]
        y = arr[start + self.lookback]
        return torch.from_numpy(X), torch.from_numpy(y), torch.tensor(d_idx, dtype=torch.long)


class GlobalLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_districts: int,
        emb_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        geo_table: torch.Tensor | None = None,
        geo_emb_dim: int = 8,
        use_id_emb: bool = True,
        post_dropout: float = 0.0,
    ):
        super().__init__()
        input_size = num_features
        self.use_id_emb = use_id_emb
        if use_id_emb:
            self.emb = nn.Embedding(num_embeddings=num_districts, embedding_dim=emb_dim)
            input_size += emb_dim
        else:
            self.emb = None

        self.has_geo = geo_table is not None
        if self.has_geo:
            # geo_table: (num_districts, geo_in_dim), fixed buffer
            self.register_buffer("geo_table", geo_table.float())
            geo_in_dim = self.geo_table.shape[1]
            self.geo_mlp = nn.Sequential(
                nn.Linear(geo_in_dim, max(geo_emb_dim, 4)),
                nn.ReLU(),
                nn.Linear(max(geo_emb_dim, 4), geo_emb_dim),
            )
            input_size += geo_emb_dim
        else:
            self.geo_table = None
            self.geo_mlp = None

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_drop = nn.Dropout(p=post_dropout) if post_dropout and post_dropout > 0 else None
        self.fc = nn.Linear(hidden_size, num_features)

    def forward(self, X: torch.Tensor, d_idx: torch.Tensor):
        # X: (B, T, F), d_idx: (B,)
        B, T, _ = X.shape
        pieces = [X]
        if self.use_id_emb and self.emb is not None:
            id_emb = self.emb(d_idx)  # (B, E)
            pieces.append(id_emb.unsqueeze(1).expand(B, T, -1))
        if self.has_geo and self.geo_table is not None and self.geo_mlp is not None:
            g = self.geo_table[d_idx]  # (B, geo_in)
            g_emb = self.geo_mlp(g)
            pieces.append(g_emb.unsqueeze(1).expand(B, T, -1))
        x_in = torch.cat(pieces, dim=-1)
        out, _ = self.lstm(x_in)
        last = out[:, -1, :]
        if self.post_drop is not None:
            last = self.post_drop(last)
        pred = self.fc(last)
        return pred


def _build_coords_from_raw(raw_csv: str = "data/hanoi_weather.csv") -> Dict[str, Tuple[float, float]]:
    # Read only needed columns for speed
    usecols = ["district", "lat", "lon"]
    try:
        df = pd.read_csv(raw_csv, usecols=usecols)
    except Exception:
        # try fallback encoding if needed
        df = pd.read_csv(raw_csv, usecols=usecols, encoding="utf-8", errors="ignore")
    df["district"] = df["district"].astype(str).map(normalize_district)
    df = df.dropna(subset=["lat", "lon"])  # ensure coords present
    coords = (
        df.groupby("district", as_index=True)[["lat", "lon"]]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    # Convert dict of dicts to plain tuple mapping
    return {k: (v["lat"], v["lon"]) for k, v in coords.items()}


def _make_geo_table(district2idx: Dict[str, int], coords: Dict[str, Tuple[float, float]], fourier_K: int = 2) -> torch.Tensor:
    # Build feature vector per district: [lat/90, lon/180, sin/cos k*lat_rad, sin/cos k*lon_rad]
    num = len(district2idx)
    feats: List[List[float]] = []
    for name, idx in sorted(district2idx.items(), key=lambda x: x[1]):
        lat, lon = coords.get(name, (0.0, 0.0))
        lat_norm = float(lat) / 90.0
        lon_norm = float(lon) / 180.0
        lat_rad = float(lat) * pi / 180.0
        lon_rad = float(lon) * pi / 180.0
        vec = [lat_norm, lon_norm]
        for k in range(1, fourier_K + 1):
            vec.extend([
                np.sin(k * lat_rad), np.cos(k * lat_rad),
                np.sin(k * lon_rad), np.cos(k * lon_rad),
            ])
        feats.append(vec)
    geo_table = torch.tensor(np.asarray(feats, dtype=np.float32))  # (num_districts, 2+4K)
    assert geo_table.shape[0] == num
    return geo_table


def build_district_index(train_csv: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv, usecols=["district"])  # small read
    districts = sorted(df["district"].astype(str).unique())
    return {name: i for i, name in enumerate(districts)}


def train_global(
    lookback: int = 24,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    emb_dim: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    geo_emb_dim: int = 8,
    fourier_K: int = 2,
    weight_decay: float = 1e-4,
    es_patience: int = 2,
    min_lr: float = 1e-5,
    lr_factor: float = 0.5,
    post_dropout: float = 0.2,
):
    train_csv = "data/splits/train.csv"
    dev_csv = "data/splits/dev.csv"
    test_csv = "data/splits/test.csv"
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)

    district2idx = build_district_index(train_csv)
    # Build geo-table from raw file coords (robust to coords csv encoding issues)
    coords = _build_coords_from_raw("data/hanoi_weather.csv")
    geo_table = _make_geo_table(district2idx, coords, fourier_K=fourier_K)
    # load combined splits
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv) if Path(dev_csv).exists() else pd.DataFrame(columns=train_df.columns)
    test_df = pd.read_csv(test_csv) if Path(test_csv).exists() else pd.DataFrame(columns=train_df.columns)

    train_ds = GlobalWeatherDataset(train_df, district2idx, lookback)
    if not train_ds.indices:
        print("No training samples found. Ensure splits exist and are non-empty.")
        return
    dev_ds = GlobalWeatherDataset(dev_df, district2idx, lookback)

    num_features = len(train_ds.feature_names)
    num_districts = len(district2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlobalLSTM(
        num_features,
        num_districts,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        geo_table=geo_table.to(device),
        geo_emb_dim=geo_emb_dim,
        use_id_emb=True,
        post_dropout=post_dropout,
    ).to(device)
    # Use weighted MSE to emphasize challenging features like precipitation
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=1, min_lr=min_lr
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    history = []
    best_dev = float("inf")
    no_improve = 0
    # Build per-feature weights (default 1.0), emphasize precipitation if present
    feat_weights = torch.ones(len(train_ds.feature_names), device=device)
    try:
        p_idx = train_ds.feature_names.index("precipitation")
        feat_weights[p_idx] = 3.0  # emphasize precipitation errors
    except ValueError:
        pass

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y, d in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            d = d.to(device)
            optimizer.zero_grad()
            pred = model(X, d)
            loss_mat = criterion(pred, y)  # (B, F)
            loss = (loss_mat * feat_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_ds)

        # dev
        model.eval()
        dev_loss_acc = 0.0
        with torch.no_grad():
            for X, y, d in dev_loader:
                X = X.float().to(device)
                y = y.float().to(device)
                d = d.to(device)
                pred = model(X, d)
                loss_mat = criterion(pred, y)
                loss = (loss_mat * feat_weights).mean()
                dev_loss_acc += loss.item() * X.size(0)
        dev_loss = dev_loss_acc / max(1, len(dev_ds))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": dev_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={dev_loss:.6f}")

        # scheduler step
        scheduler.step(dev_loss)

        if dev_loss < best_dev - 1e-6:
            best_dev = dev_loss
            no_improve = 0
            torch.save(model.state_dict(), model_dir / "global_lstm.pt")
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"Early stopping at epoch {epoch} (best val={best_dev:.6f})")
                break

    # save history
    pd.DataFrame(history).to_csv(model_dir / "global_loss.csv", index=False)

    # save config
    cfg = {
        "lookback": lookback,
        "num_features": num_features,
        "num_districts": num_districts,
        "emb_dim": emb_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "geo_emb_dim": geo_emb_dim,
        "fourier_K": fourier_K,
        "weight_decay": weight_decay,
        "es_patience": es_patience,
        "min_lr": min_lr,
        "lr_factor": lr_factor,
        "post_dropout": post_dropout,
        "district2idx": district2idx,
        "feature_names": train_ds.feature_names,
        "geo_features": "lat_norm,lon_norm + sincos(k*lat, k*lon), k=1..K",
    }
    with open(model_dir / "global_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # quick evaluation on test set
    evaluate_global(model_dir / "global_lstm.pt", model_dir / "global_config.json", test_df)


def evaluate_global(model_path: Path, config_path: Path, test_df: pd.DataFrame | str):
    if not Path(model_path).exists() or not Path(config_path).exists():
        print("Global model not found; skip evaluation.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback = int(cfg["lookback"])
    district2idx = {k: int(v) for k, v in cfg["district2idx"].items()}
    feature_names = cfg["feature_names"]
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = _build_coords_from_raw("data/hanoi_weather.csv")
    geo_table = _make_geo_table(district2idx, coords, fourier_K=int(cfg.get("fourier_K", 2)))
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

    # Load combined test DataFrame if a path was provided
    if isinstance(test_df, (str, Path)):
        test_df = pd.read_csv(test_df)

    results = []
    # Accumulators for overall (micro) metrics
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_count = 0  # number of time steps (rows) aggregated across districts
    sum_y_vec = None  # per-feature sums
    sum_y2_vec = None  # per-feature squared sums
    for district, g in test_df.groupby("district", sort=False):
        if district not in district2idx:
            continue
        d_idx = district2idx[district]
        feat_df = g.select_dtypes(include=[float, int, np.number])
        if feat_df.empty or len(feat_df) <= lookback:
            continue
        values = feat_df.values.astype(np.float32)
        X_list, y_list = [], []
        for start in range(0, len(values) - lookback):
            X_list.append(values[start : start + lookback])
            y_list.append(values[start + lookback])
        X = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
        y = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
        d = torch.full((X.shape[0],), d_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(X, d)
        mae = torch.mean(torch.abs(pred - y)).item()
        rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
        # R^2 (coefficient of determination), averaged across features
        # Handle edge case when variance is zero (ss_tot == 0)
        y_mean = torch.mean(y, dim=0, keepdim=True)
        ss_res = torch.sum((pred - y) ** 2)
        ss_tot = torch.sum((y - y_mean) ** 2)
        if ss_tot.item() == 0:
            # If predictions equal targets exactly, r2=1.0, else 0.0
            r2 = 1.0 if torch.allclose(pred, y) else 0.0
        else:
            r2 = (1.0 - (ss_res / ss_tot)).item()
        results.append({"district": district, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"Global eval - {district}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        # Update overall accumulators (micro)
        err = pred - y
        total_abs_err += torch.sum(torch.abs(err)).item()
        total_sq_err += torch.sum(err * err).item()
        total_count += y.shape[0]
        y_sum = torch.sum(y, dim=0).detach().cpu().numpy()
        y_sum2 = torch.sum(y * y, dim=0).detach().cpu().numpy()
        if sum_y_vec is None:
            sum_y_vec = y_sum
            sum_y2_vec = y_sum2
        else:
            sum_y_vec += y_sum
            sum_y2_vec += y_sum2

    out = pd.DataFrame(results)
    out_path = Path("model/global_eval.csv")
    out.to_csv(out_path, index=False)
    print(f"Global evaluation saved -> {out_path}")

    # Compute aggregate metrics (micro and macro) and save
    try:
        # Micro (over all samples)
        denom = max(1, total_count * num_features)
        micro_mae = total_abs_err / denom
        micro_rmse = float(np.sqrt(total_sq_err / denom))
        # R2 micro
        if total_count > 0 and sum_y_vec is not None and sum_y2_vec is not None:
            mean_vec = sum_y_vec / float(total_count)
            ss_tot_total = float(np.sum(sum_y2_vec - float(total_count) * (mean_vec ** 2)))
            if ss_tot_total == 0.0:
                micro_r2 = 1.0 if total_sq_err == 0.0 else 0.0
            else:
                micro_r2 = 1.0 - (total_sq_err / ss_tot_total)
        else:
            micro_r2 = float("nan")

        # Macro (mean of per-district metrics)
        if not out.empty:
            macro_mae = float(out["MAE"].mean())
            macro_rmse = float(out["RMSE"].mean())
            macro_r2 = float(out["R2"].mean())
        else:
            macro_mae = macro_rmse = macro_r2 = float("nan")

        overall_df = pd.DataFrame([
            {"scope": "micro", "MAE": micro_mae, "RMSE": micro_rmse, "R2": micro_r2},
            {"scope": "macro", "MAE": macro_mae, "RMSE": macro_rmse, "R2": macro_r2},
        ])
        overall_path = Path("model/global_eval_overall.csv")
        overall_df.to_csv(overall_path, index=False)
        print(f"Global overall metrics saved -> {overall_path}")
    except Exception as e:
        print(f"Failed to compute/save overall metrics: {e}")


def run_train():
    """Entry for main.py to train the global model."""
    train_global()


if __name__ == "__main__":
    train_global()
