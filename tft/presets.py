import os
from pathlib import Path

from preprocess import run_preprocess, run_preprocess_daily
from train_tft import train_tft


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip() in ("1", "true", "True")


def _should_run_preprocess() -> bool:
    if _truthy("SKIP_PREPROCESS"):
        print("SKIP_PREPROCESS=1 -> skip preprocessing.")
        return False
    if _truthy("FORCE_PREPROCESS"):
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


def run_train_presets():
    # Ensure base (hourly) splits
    if _should_run_preprocess():
        run_preprocess()
    else:
        print("Skip preprocessing (outputs present and up-to-date).")

    # Train 12h ahead on hourly data
    lb_12 = int(os.environ.get("LOOKBACK_H12", "48"))
    print(f"[Preset] Train 12h ahead, lookback={lb_12}")
    train_tft(lookback=lb_12, horizon=12, splits_dir="data/splits", run_name="h12h")

    # Train 24h ahead on hourly data
    lb_24 = int(os.environ.get("LOOKBACK_H24", "72"))
    print(f"[Preset] Train 24h ahead, lookback={lb_24}")
    train_tft(lookback=lb_24, horizon=24, splits_dir="data/splits", run_name="h24h")

    # Daily 7-day ahead: build daily splits then train with horizon=7
    print("[Preset] Prepare daily splits for 7-day forecasting")
    run_preprocess_daily()
    lb_d7 = int(os.environ.get("LOOKBACK_D7", "30"))
    print(f"[Preset] Train daily 7-day ahead, lookback={lb_d7}")
    train_tft(lookback=lb_d7, horizon=7, splits_dir="data/splits_daily", run_name="daily7d")


if __name__ == "__main__":
    run_train_presets()

