import os
from pathlib import Path

from preprocess import run_preprocess
from train_tft import train_tft
from evaluate import compute_overall


def _should_run_preprocess() -> bool:
    if os.environ.get("SKIP_PREPROCESS", "").strip() in ("1", "true", "True"):
        print("SKIP_PREPROCESS=1 -> skip preprocessing.")
        return False
    if os.environ.get("FORCE_PREPROCESS", "").strip() in ("1", "true", "True"):
        print("FORCE_PREPROCESS=1 -> run preprocessing.")
        return True

    base_root = Path(__file__).resolve().parents[1]
    input_csv = base_root / "data/hanoi_weather.csv"
    out_dir = base_root / "data/splits"
    required = [out_dir / "train.csv", out_dir / "dev.csv", out_dir / "test.csv"]

    if not all(p.exists() for p in required):
        print("Preprocess outputs missing -> run preprocessing.")
        return True
    if not input_csv.exists():
        print("data/hanoi_weather.csv not found; using existing splits.")
        return False

    try:
        in_mtime = input_csv.stat().st_mtime
        out_mtimes = [p.stat().st_mtime for p in required]
        if any(m < in_mtime for m in out_mtimes):
            print("Input CSV is newer than splits -> run preprocessing.")
            return True
    except Exception:
        return True
    return False


def _run_variant(run_name: str, lookback: int, horizon: int, d_model: int, nhead: int, num_layers: int, dropout: float, weight_decay: float, lr: float | None = None, es_patience: int | None = None):
    kw = dict(
        lookback=lookback,
        horizon=horizon,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        weight_decay=weight_decay,
        run_name=run_name,
    )
    if lr is not None:
        kw["lr"] = lr
    if es_patience is not None:
        kw["es_patience"] = es_patience
    train_tft(**kw)
    model_dir = str(Path(__file__).resolve().parent / f"model/{run_name}")
    os.environ["MODEL_DIR"] = model_dir
    compute_overall(model_dir)


if __name__ == "__main__":
    # Apply log1p transform to precipitation-like feature during preprocessing
    os.environ["LOG1P_RAIN"] = "1"
    if _should_run_preprocess():
        run_preprocess()
    else:
        print("Skip preprocessing (outputs present and up-to-date).")

    # 6h
    _run_variant("h6h", lookback=36, horizon=6, d_model=160, nhead=4, num_layers=2, dropout=0.15, weight_decay=5e-4)
    # 12h
    _run_variant("h12h", lookback=48, horizon=12, d_model=192, nhead=4, num_layers=3, dropout=0.1, weight_decay=1e-4)
    # 24h (stronger regularization to mitigate overfit)
    _run_variant("h24h", lookback=72, horizon=24, d_model=160, nhead=4, num_layers=3, dropout=0.3, weight_decay=1e-3, lr=8e-4, es_patience=5)
    # 7d (daily)
    _run_variant("daily7d", lookback=30, horizon=7, d_model=192, nhead=4, num_layers=3, dropout=0.15, weight_decay=5e-4)
