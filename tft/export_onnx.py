import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_tft import TFTLight  # type: ignore
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_fallback_geo_table(num_districts: int, fourier_K: int) -> torch.Tensor:
    geo_dim = 2 + 4 * int(fourier_K)
    return torch.zeros((int(num_districts), geo_dim), dtype=torch.float32)


def _mk_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    tmp = df.copy()
    dt = pd.to_datetime(tmp["datetime"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(int)
    dow = dt.dt.dayofweek.fillna(0).astype(int)
    mon = dt.dt.month.fillna(1).astype(int)
    tmp["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    tmp["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    tmp["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    tmp["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    tmp["month_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    tmp["month_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    return tmp


def _train_slice_per_district(df: pd.DataFrame, train_ratio: float = 0.7) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("district", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g)
        end = int(n * train_ratio)
        parts.append(g.iloc[:end])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def compute_minmax_from_raw(raw_csv: Path, feature_names: List[str], add_time_feats: bool, log1p_precip: bool,
                            precip_name: Optional[str], train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(raw_csv, parse_dates=["datetime"]) if raw_csv.exists() else pd.read_csv(raw_csv)
    df["district"] = df["district"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    if add_time_feats:
        df = _mk_time_feats(df)
    if log1p_precip and precip_name and precip_name in df.columns:
        vals = df[precip_name].astype(float).to_numpy()
        vals = np.clip(vals, a_min=0.0, a_max=None)
        df.loc[:, precip_name] = np.log1p(vals)
    train_df = _train_slice_per_district(df, train_ratio=train_ratio)
    X = train_df[feature_names].astype(float)
    mins = X.min(axis=0).to_numpy()
    maxs = X.max(axis=0).to_numpy()
    return mins, maxs


class TFTWithInverseScale(nn.Module):
    def __init__(self, base: nn.Module, mins: torch.Tensor, maxs: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("mins", mins.view(1, 1, -1).float())
        self.register_buffer("maxs", maxs.view(1, 1, -1).float())

    def forward(self, X: torch.Tensor, d_idx: torch.Tensor) -> torch.Tensor:
        y = self.base(X, d_idx)
        return y * (self.maxs - self.mins) + self.mins


class TFTWithInverseScaleAndExp(nn.Module):
    def __init__(self, base: nn.Module, mins: torch.Tensor, maxs: torch.Tensor, precip_idx: int):
        super().__init__()
        self.base = base
        self.register_buffer("mins", mins.view(1, 1, -1).float())
        self.register_buffer("maxs", maxs.view(1, 1, -1).float())
        self.precip_idx = int(precip_idx)

    def forward(self, X: torch.Tensor, d_idx: torch.Tensor) -> torch.Tensor:
        y = self.base(X, d_idx)
        y = y * (self.maxs - self.mins) + self.mins
        if self.precip_idx >= 0:
            p = y[..., self.precip_idx]
            p = torch.clamp(p, min=0.0)
            p = torch.exp(p) - 1.0
            y = torch.cat([y[..., : self.precip_idx], p.unsqueeze(-1), y[..., self.precip_idx + 1 :]], dim=-1)
        return y


def export_onnx(model_dir: Path, onnx_out: Optional[Path] = None, *,
                unscale: bool = True, raw_csv: Optional[Path] = None,
                train_ratio: float = 0.7, log1p_precip: bool = False) -> Path:
    model_dir = model_dir.resolve()
    config_path = model_dir / "global_config.json"
    model_path = model_dir / "global_tft.pt"
    if onnx_out is None:
        onnx_out = model_dir / "global_tft.onnx"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model or config in {model_dir}")

    cfg = _load_config(config_path)
    lookback = int(cfg["lookback"])
    horizon = int(cfg.get("horizon", 1))
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])
    feature_names = list(cfg.get("feature_names", []))

    model = TFTLight(
        num_features=num_features,
        num_districts=num_districts,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        geo_table=_make_fallback_geo_table(num_districts, int(cfg.get("fourier_K", 2))),
        geo_emb_dim=int(cfg.get("geo_emb_dim", 8)),
        id_emb_dim=int(cfg.get("id_emb_dim", 8)),
        horizon=horizon,
    )
    device = torch.device("cpu")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Optionally wrap with inverse MinMax scaling
    if unscale:
        raw_csv = raw_csv or Path(__file__).resolve().parents[1] / "data/hanoi_weather.csv"
        precip_name = "precipitation" if "precipitation" in feature_names else ("rain" if "rain" in feature_names else None)
        mins_np, maxs_np = compute_minmax_from_raw(raw_csv, feature_names, add_time_feats=True,
                                                   log1p_precip=log1p_precip, precip_name=precip_name,
                                                   train_ratio=train_ratio)
        if log1p_precip and precip_name in feature_names:
            p_idx = feature_names.index(precip_name)
            model_export = TFTWithInverseScaleAndExp(model, torch.from_numpy(mins_np), torch.from_numpy(maxs_np), p_idx)
        else:
            model_export = TFTWithInverseScale(model, torch.from_numpy(mins_np), torch.from_numpy(maxs_np))
    else:
        model_export = model

    X_dummy = torch.zeros((1, lookback, num_features), dtype=torch.float32)
    d_dummy = torch.zeros((1,), dtype=torch.long)

    torch.onnx.export(
        model_export,
        (X_dummy, d_dummy),
        str(onnx_out),
        input_names=["X", "district_idx"],
        output_names=["y_pred"],
        dynamic_axes={
            "X": {0: "batch", 1: "time"},
            "district_idx": {0: "batch"},
            "y_pred": {0: "batch", 1: "horizon"},
        },
        opset_version=18,
    )
    return onnx_out


def main():
    parser = argparse.ArgumentParser(description="Export TFT-Light model to ONNX (optionally unscale outputs)")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).resolve().parent / "model"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-unscale", action="store_true", help="Do not inverse MinMax on outputs")
    parser.add_argument("--raw-csv", type=str, default=None, help="Path to raw CSV to compute MinMax (default: data/hanoi_weather.csv)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio per district when computing MinMax")
    parser.add_argument("--log1p-precip", action="store_true", help="Apply log1p to precipitation/rain when computing MinMax (must match training)")
    args = parser.parse_args()

    out = export_onnx(
        Path(args.model_dir),
        Path(args.out) if args.out else None,
        unscale=(not args.no_unscale),
        raw_csv=(Path(args.raw_csv) if args.raw_csv else None),
        train_ratio=float(args.train_ratio),
        log1p_precip=bool(args.log1p_precip),
    )
    print(f"Exported ONNX -> {out}")


if __name__ == "__main__":
    main()
