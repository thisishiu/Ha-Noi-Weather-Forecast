from pathlib import Path
from train_tft import evaluate_global


def main():
    base_dir = Path(__file__).resolve().parent
    base_root = base_dir.parents[0]
    model_path = base_dir / "model/global_tft.pt"
    config_path = base_dir / "model/global_config.json"
    test_csv = base_root / "data/splits/test.csv"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError("Model or config not found. Train TFT first.")
    # Pass through even if existence check is flaky in some environments
    evaluate_global(model_path, config_path, str(test_csv))


if __name__ == "__main__":
    main()
