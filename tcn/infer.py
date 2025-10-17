import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from train_tcn import GlobalTCN  # type: ignore


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_fallback_geo_table(num_districts: int, fourier_K: int) -> torch.Tensor:
    geo_dim = 2 + 4 * int(fourier_K)
    return torch.zeros((int(num_districts), geo_dim), dtype=torch.float32)


@torch.no_grad()
def forecast_k_steps(model: GlobalTCN, X_window: torch.Tensor, d_idx: int, k: int) -> List[torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    device = next(model.parameters()).device
    x = X_window.clone().to(device)
    d = torch.tensor([d_idx], dtype=torch.long, device=device)
    lookback = x.shape[0]
    for _ in range(int(k)):
        inp = x.unsqueeze(0)  # [1, T, F]
        y = model(inp, d)  # [1,H,F] or [1,F]
        if y.dim() == 3:
            y_step = y[:, 0, :].squeeze(0)
        else:
            y_step = y.squeeze(0)
        preds.append(y_step.detach().cpu())
        if lookback > 0:
            x = torch.cat([x[1:], y_step.unsqueeze(0)], dim=0)
        else:
            x = y_step.unsqueeze(0)
    return preds


def main():
    parser = argparse.ArgumentParser(description="Iterative multi-step forecast with TCN model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "model"),
        help="Directory containing global_tcn.pt and global_config.json",
    )
    parser.add_argument("--district", type=str, required=True, help="District name in config mapping")
    parser.add_argument("--window-npy", type=str, required=True, help="Path to .npy of shape [T,F] (last lookback steps)")
    parser.add_argument("--steps", type=int, default=24, help="Number of steps to forecast")
    parser.add_argument("--out", type=str, default=None, help="Output .npy for forecasts [K,F]")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = _load_config(model_dir / "global_config.json")
    num_features = int(cfg["num_features"])
    num_districts = int(cfg["num_districts"])
    d2i = {k: int(v) for k, v in cfg["district2idx"].items()}
    if args.district not in d2i:
        raise ValueError(f"District '{args.district}' not found in config")

    model = GlobalTCN(
        num_features=num_features,
        num_districts=num_districts,
        emb_dim=int(cfg.get("emb_dim", 8)),
        channels=int(cfg.get("tcn_channels", 64)),
        num_blocks=int(cfg.get("tcn_num_blocks", 4)),
        kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        dropout=float(cfg.get("tcn_dropout", 0.1)),
        geo_table=_make_fallback_geo_table(num_districts, int(cfg.get("fourier_K", 2))),
        geo_emb_dim=int(cfg.get("geo_emb_dim", 8)),
        use_id_emb=True,
        post_dropout=float(cfg.get("post_dropout", 0.0)),
    )
    state = torch.load(model_dir / "global_tcn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X_np = np.load(args.window_npy)
    if X_np.ndim != 2 or X_np.shape[1] != num_features:
        raise ValueError(f"window-npy must have shape [T,{num_features}], got {X_np.shape}")
    lookback = int(cfg["lookback"]) if "lookback" in cfg else X_np.shape[0]
    if X_np.shape[0] < lookback:
        raise ValueError(f"window has T={X_np.shape[0]} < lookback={lookback}")
    X_win = torch.tensor(X_np[-lookback:], dtype=torch.float32)

    preds = forecast_k_steps(model, X_win, d2i[args.district], int(args.steps))
    Y = torch.stack(preds, dim=0).numpy()
    out_path = Path(args.out) if args.out else (model_dir / f"forecast_{args.district}_{args.steps}.npy")
    np.save(out_path, Y)
    print(f"Saved forecasts -> {out_path} (shape={Y.shape})")


if __name__ == "__main__":
    main()
