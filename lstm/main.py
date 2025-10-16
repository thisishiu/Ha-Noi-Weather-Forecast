import os
from pathlib import Path

from preprocess import run_preprocess
from train_global import run_train
from evaluate import run_evaluate
from visualize import run_visualize


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


if __name__ == "__main__":
    if _should_run_preprocess():
        run_preprocess()
    else:
        print("Skip preprocessing (outputs present and up-to-date).")
    run_train()
    run_evaluate()
    run_visualize()
