# src/main.py
import os
from pathlib import Path

from preprocess import run_preprocess
from train_global import run_train
from evaluate import run_evaluate
from visualize import run_visualize


def _should_run_preprocess() -> bool:
    """Decide whether to run preprocessing.

    Rules:
    - SKIP_PREPROCESS=1 -> always skip
    - FORCE_PREPROCESS=1 -> always run
    - Otherwise: run only if outputs are missing or older than input CSV
    """
    if os.environ.get("SKIP_PREPROCESS", "").strip() in ("1", "true", "True"):
        print("SKIP_PREPROCESS=1 -> skip preprocessing.")
        return False
    if os.environ.get("FORCE_PREPROCESS", "").strip() in ("1", "true", "True"):
        print("FORCE_PREPROCESS=1 -> run preprocessing.")
        return True

    input_csv = Path("data/hanoi_weather.csv")
    out_dir = Path("data/splits")
    required = [out_dir / "train.csv", out_dir / "dev.csv", out_dir / "test.csv"]

    # Any output missing -> run
    if not all(p.exists() for p in required):
        print("Preprocess outputs missing -> run preprocessing.")
        return True

    # If input missing, we keep existing outputs
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
        # On any error determining mtimes, be conservative and run
        return True

    # Outputs exist and are up-to-date
    return False


if __name__ == "__main__":
    # 1) Preprocess only when needed
    if _should_run_preprocess():
        run_preprocess()
    else:
        print("Skip preprocessing (outputs present and up-to-date).")

    # 2) Train global model
    run_train()

    # 3) Evaluate per district
    run_evaluate()

    # 4) Visualize results
    run_visualize()

