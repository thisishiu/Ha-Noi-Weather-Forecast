import json
import pandas as pd
from pathlib import Path


def evaluate_all():
    base_dir = Path(__file__).resolve().parent
    global_path = base_dir / "model/global_eval.csv"
    global_overall_path = base_dir / "model/global_eval_overall.csv"

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
        cfg_path = base_dir / "model/global_config.json"
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
        (base_dir / "model").mkdir(parents=True, exist_ok=True)
        df.to_csv(base_dir / "model/eval_results.csv", index=False)
        print(f"Evaluation summary saved -> {base_dir / 'model/eval_results.csv'}")
        return

    print("No evaluation files found. Run training first.")


def run_evaluate():
    evaluate_all()


if __name__ == "__main__":
    evaluate_all()

