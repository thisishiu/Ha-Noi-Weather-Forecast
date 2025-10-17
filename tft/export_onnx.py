import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from train_tft import TFTLight  # type: ignore


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_fallback_geo_table(num_districts: int, fourier_K: int) -> torch.Tensor:
    geo_dim = 2 + 4 * int(fourier_K)
    return torch.zeros((int(num_districts), geo_dim), dtype=torch.float32)


def export_onnx(model_dir: Path, onnx_out: Optional[Path] = None) -> Path:
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

    X_dummy = torch.zeros((1, lookback, num_features), dtype=torch.float32)
    d_dummy = torch.zeros((1,), dtype=torch.long)

    torch.onnx.export(
        model,
        (X_dummy, d_dummy),
        str(onnx_out),
        input_names=["X", "district_idx"],
        output_names=["y_pred"],
        dynamic_axes={
            "X": {0: "batch", 1: "time"},
            "district_idx": {0: "batch"},
            "y_pred": {0: "batch", 1: "horizon"},
        },
        opset_version=17,
    )
    return onnx_out


def main():
    parser = argparse.ArgumentParser(description="Export TFT-Light model to ONNX")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "model"),
        help="Directory containing global_tft.pt and global_config.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output ONNX path (default: <model-dir>/global_tft.onnx)",
    )
    args = parser.parse_args()
    out = export_onnx(Path(args.model_dir), Path(args.out) if args.out else None)
    print(f"Exported ONNX -> {out}")


if __name__ == "__main__":
    main()

