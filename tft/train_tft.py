import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from math import pi
from torch.amp import autocast, GradScaler

from utils import normalize_district


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    tmp = df.copy()
    dt = pd.to_datetime(tmp["datetime"], errors="coerce")
    # Calendar features (sin/cos) bounded and scale-free
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

class GlobalWeatherDataset(Dataset):
    def __init__(self, df: pd.DataFrame, district2idx: Dict[str, int], lookback: int = 48, horizon: int = 6):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.district2idx = district2idx
        self.series: Dict[int, np.ndarray] = {}
        self.feature_names: List[str] = []
        self.indices: List[Tuple[int, int]] = []

        if df is None or df.empty:
            return
        feat_df = df.select_dtypes(include=[float, int, np.number])
        self.feature_names = list(feat_df.columns)
        for dname, g in df.groupby("district", sort=False):
            if dname not in district2idx:
                continue
            d_idx = district2idx[dname]
            vals = g[self.feature_names].values.astype(np.float32)
            self.series[d_idx] = vals
            min_len = self.lookback + self.horizon
            if len(vals) >= min_len:
                for start in range(0, len(vals) - min_len + 1):
                    self.indices.append((d_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, start = self.indices[idx]
        arr = self.series[d_idx]
        X = arr[start : start + self.lookback]
        y = arr[start + self.lookback : start + self.lookback + self.horizon]
        return torch.from_numpy(X), torch.from_numpy(y), torch.tensor(d_idx, dtype=torch.long)


def _build_coords_from_raw(raw_csv: str = "data/hanoi_weather.csv") -> Dict[str, Tuple[float, float]]:
    usecols = ["district", "lat", "lon"]
    try:
        df = pd.read_csv(raw_csv, usecols=usecols)
    except Exception:
        df = pd.read_csv(raw_csv, usecols=usecols, encoding="utf-8", errors="ignore")
    df["district"] = df["district"].astype(str).map(normalize_district)
    df = df.dropna(subset=["lat", "lon"])
    coords = (
        df.groupby("district", as_index=True)[["lat", "lon"]]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    return {k: (v["lat"], v["lon"]) for k, v in coords.items()}


def _make_geo_table(district2idx: Dict[str, int], coords: Dict[str, Tuple[float, float]], fourier_K: int = 2) -> torch.Tensor:
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
    geo_table = torch.tensor(np.asarray(feats, dtype=np.float32))
    assert geo_table.shape[0] == num
    return geo_table


def build_district_index(train_csv: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv, usecols=["district"])  # small read
    districts = sorted(df["district"].astype(str).unique())
    return {name: i for i, name in enumerate(districts)}


class TFTLight(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_districts: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        geo_table: torch.Tensor | None = None,
        geo_emb_dim: int = 8,
        id_emb_dim: int = 8,
        horizon: int = 6,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.use_geo = geo_table is not None
        self.id_emb = nn.Embedding(num_embeddings=num_districts, embedding_dim=id_emb_dim)
        in_dim = num_features + id_emb_dim
        if self.use_geo:
            self.register_buffer("geo_table", geo_table.float())
            self.geo_mlp = nn.Sequential(
                nn.Linear(self.geo_table.shape[1], max(geo_emb_dim, 4)),
                nn.ReLU(),
                nn.Linear(max(geo_emb_dim, 4), geo_emb_dim),
            )
            in_dim += geo_emb_dim
        else:
            self.geo_mlp = None

        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        self.horizon_pos = nn.Embedding(self.horizon, d_model)
        self.post = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_features)
        )

    def forward(self, X: torch.Tensor, d_idx: torch.Tensor) -> torch.Tensor:
        # X: [B,T,F]
        B, T, F = X.shape
        id_emb = self.id_emb(d_idx)  # [B, D]
        id_rep = id_emb.unsqueeze(1).expand(B, T, -1)
        pieces = [X, id_rep]
        if self.use_geo and self.geo_mlp is not None:
            g = self.geo_table[d_idx]
            gemb = self.geo_mlp(g).unsqueeze(1).expand(B, T, -1)
            pieces.append(gemb)
        xin = torch.cat(pieces, dim=-1)
        h = self.input_proj(xin)  # [B,T,d]
        h_enc = self.encoder(h)  # [B,T,d]
        h_enc = self.enc_norm(h_enc)
        last = h_enc[:, -1, :]  # [B,d]
        # Repeat for horizon and add positional horizon embedding
        H = self.horizon
        last_rep = last.unsqueeze(1).expand(B, H, -1)  # [B,H,d]
        pos = torch.arange(H, device=X.device)
        hp = last_rep + self.horizon_pos(pos).unsqueeze(0)
        out = self.post(hp)  # [B,H,F]
        return out


def train_tft(
    lookback: int = 48,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    d_model: int = 192,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
    geo_emb_dim: int = 8,
    fourier_K: int = 2,
    weight_decay: float = 1e-4,
    es_patience: int = 8,
    min_lr: float = 1e-5,
    lr_factor: float = 0.5,
    horizon: int = 6,
    id_emb_dim: int = 8,
    huber_beta_env: str = "HUBER_BETA",
    scheduler_type: str = "onecycle",
    horizon_gamma: float = 0.5,
):
    base_root = Path(__file__).resolve().parents[1]
    train_csv = str(base_root / "data/splits/train.csv")
    dev_csv = str(base_root / "data/splits/dev.csv")
    test_csv = str(base_root / "data/splits/test.csv")
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    district2idx = build_district_index(train_csv)
    coords = _build_coords_from_raw(str(base_root / "data/hanoi_weather.csv"))
    geo_table = _make_geo_table(district2idx, coords, fourier_K=fourier_K)

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv) if Path(dev_csv).exists() else pd.DataFrame(columns=train_df.columns)
    test_df = pd.read_csv(test_csv) if Path(test_csv).exists() else pd.DataFrame(columns=train_df.columns)

    # Add calendar covariates to all splits (numeric, no scaling required)
    train_df = _add_time_features(train_df)
    if not dev_df.empty:
        dev_df = _add_time_features(dev_df)
    if not test_df.empty:
        test_df = _add_time_features(test_df)

    train_ds = GlobalWeatherDataset(train_df, district2idx, lookback, horizon)
    if not train_ds.indices:
        print("No training samples found. Ensure splits exist and are non-empty.")
        return
    dev_ds = GlobalWeatherDataset(dev_df, district2idx, lookback, horizon)

    num_features = len(train_ds.feature_names)
    num_districts = len(district2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TFTLight(
        num_features=num_features,
        num_districts=num_districts,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        geo_table=geo_table.to(device),
        geo_emb_dim=geo_emb_dim,
        id_emb_dim=id_emb_dim,
        horizon=horizon,
    ).to(device)

    def _huber_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        err = pred - target
        abs_err = torch.abs(err)
        return torch.where(abs_err <= beta, 0.5 * (err * err) / beta, abs_err - 0.5 * beta)

    try:
        huber_beta = float(os.environ.get(huber_beta_env, "0.5"))
    except Exception:
        huber_beta = 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Scheduler placeholder; if OneCycle, rebuild after we know steps_per_epoch
    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=1, min_lr=min_lr
        )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if scheduler_type == "onecycle":
        steps = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

    history = []
    best_dev = float("inf")
    no_improve = 0
    # Uniform per-feature weighting (no rain-specific weighting)

    H = int(horizon)
    h_idx = torch.arange(H, device=device, dtype=torch.float32)
    h_w = torch.pow(torch.tensor(float(horizon_gamma), device=device), h_idx)
    h_w = h_w / torch.mean(h_w)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y, d in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            d = d.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(X, d)
                loss_mat = _huber_loss(pred, y, beta=huber_beta)
                loss = (loss_mat * h_w.view(1, H, 1)).mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if scheduler_type == "onecycle":
                try:
                    scheduler.step()
                except Exception:
                    pass
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        dev_loss_acc = 0.0
        with torch.no_grad():
            for X, y, d in dev_loader:
                X = X.float().to(device)
                y = y.float().to(device)
                d = d.to(device)
                pred = model(X, d)
                loss_mat = _huber_loss(pred, y, beta=huber_beta)
                loss = (loss_mat * h_w.view(1, H, 1)).mean()
                dev_loss_acc += loss.item() * X.size(0)
        dev_loss = dev_loss_acc / max(1, len(dev_ds))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": dev_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={dev_loss:.6f}")

        if scheduler_type != "onecycle":
            scheduler.step(dev_loss)

        if dev_loss < best_dev - 1e-6:
            best_dev = dev_loss
            no_improve = 0
            torch.save(model.state_dict(), model_dir / "global_tft.pt")
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"Early stopping at epoch {epoch} (best val={best_dev:.6f})")
                break

    pd.DataFrame(history).to_csv(model_dir / "global_loss.csv", index=False)

    cfg = {
        "lookback": lookback,
        "horizon": horizon,
        "num_features": num_features,
        "num_districts": num_districts,
        "model_type": "TFT-Light",
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "geo_emb_dim": geo_emb_dim,
        "fourier_K": fourier_K,
        "weight_decay": weight_decay,
        "es_patience": es_patience,
        "min_lr": min_lr,
        "lr_factor": lr_factor,
        "id_emb_dim": id_emb_dim,
        "district2idx": district2idx,
        "feature_names": train_ds.feature_names,
        "geo_features": "lat_norm,lon_norm + sincos(k*lat, k*lon), k=1..K",
    }
    with open(model_dir / "global_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    evaluate_global(model_dir / "global_tft.pt", model_dir / "global_config.json", test_df)


def evaluate_global(model_path: Path, config_path: Path, test_df: pd.DataFrame | str):
    if not Path(model_path).exists() or not Path(config_path).exists():
        print("Global model not found; skip evaluation.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lookback = int(cfg["lookback"])
    horizon = int(cfg.get("horizon", 1))
    district2idx = {k: int(v) for k, v in cfg["district2idx"].items()}
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_root = Path(__file__).resolve().parents[1]
    coords = _build_coords_from_raw(str(base_root / "data/hanoi_weather.csv"))
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if isinstance(test_df, (str, Path)):
        test_df = pd.read_csv(test_df)
    # Recreate time features for evaluation consistency
    test_df = _add_time_features(test_df)

    results = []
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
        # Ensure same feature ordering as training
        feature_names = list(cfg.get("feature_names", []))
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
        mae = torch.mean(torch.abs(pred - y)).item()
        rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
        y_mean = torch.mean(y, dim=0, keepdim=True)
        ss_res = torch.sum((pred - y) ** 2)
        ss_tot = torch.sum((y - y_mean) ** 2)
        if ss_tot.item() == 0:
            r2 = 1.0 if torch.allclose(pred, y) else 0.0
        else:
            r2 = (1.0 - (ss_res / ss_tot)).item()
        results.append({"district": district, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"Global eval - {district}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

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

    base_dir = Path(__file__).resolve().parent
    out = pd.DataFrame(results)
    out_path = base_dir / "model/global_eval.csv"
    out.to_csv(out_path, index=False)
    print(f"Global evaluation saved -> {out_path}")

    try:
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
        overall_path = base_dir / "model/global_eval_overall.csv"
        overall_df.to_csv(overall_path, index=False)
        print(f"Global overall metrics saved -> {overall_path}")
    except Exception as e:
        print(f"Failed to compute/save overall metrics: {e}")


def run_train_tft():
    train_tft()


if __name__ == "__main__":
    train_tft()
