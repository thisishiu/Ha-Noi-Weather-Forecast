import json
import os
import pandas as pd
from pathlib import Path


def _resolve_model_dir(model_dir: str | None) -> Path:
    base_dir = Path(__file__).resolve().parent
    if model_dir:
        p = Path(model_dir)
    else:
        env_dir = os.environ.get("MODEL_DIR", "").strip()
        p = Path(env_dir) if env_dir else (base_dir / "model")
    return p.resolve()


def evaluate_all(model_dir: str | None = None):
    model_root = _resolve_model_dir(model_dir)
    global_path = model_root / "global_eval.csv"
    global_overall_path = model_root / "global_eval_overall.csv"

    if global_path.exists():
        try:
            df = pd.read_csv(global_path)
        except Exception:
            print("Global evaluation file is empty or unreadable.")
            return
        if df.empty:
            print("Global evaluation file is empty.")
            return
        model_type = None
        cfg_path = model_root / "global_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                model_type = str(cfg.get("model_type", "")).upper() or None
            except Exception:
                model_type = None
        header = "Evaluation results (Global model)" if not model_type else f"Evaluation results (Global {model_type})"
        print(header + ":")

        if global_overall_path.exists():
            try:
                df_overall = pd.read_csv(global_overall_path)
                if not df_overall.empty and {"scope", "MAE", "RMSE"}.issubset(df_overall.columns):
                    print("Overall metrics:")
                    for _, row in df_overall.iterrows():
                        scope = row.get("scope", "?")
                        mae_o = row.get("MAE", float("nan"))
                        rmse_o = row.get("RMSE", float("nan"))
                        r2_o = row.get("R2", float("nan"))
                        print(f"  {scope}: MAE={mae_o:.4f}, RMSE={rmse_o:.4f}, R2={r2_o:.4f}")
            except Exception:
                pass
        has_r2 = "R2" in df.columns
        for _, row in df.iterrows():
            district = row.get("district", "?")
            mae = row.get("MAE", float("nan"))
            rmse = row.get("RMSE", float("nan"))
            if has_r2:
                r2 = row.get("R2", float("nan"))
                print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            else:
                print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        model_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(model_root / "eval_results.csv", index=False)
        print(f"Evaluation summary saved -> {model_root / 'eval_results.csv'}")
        return

    print("No evaluation files found. Run training first.")


def run_evaluate(model_dir: str | None = None):
    evaluate_all(model_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print evaluation results for a model directory")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model directory")
    args = parser.parse_args()
    evaluate_all(args.model_dir)
