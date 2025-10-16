import pandas as pd
from pathlib import Path


def evaluate_all():
    """Print evaluation results for the best available pipeline.

    Priority:
    1) Global model results at `model/global_eval.csv`
    2) Hybrid per-district results at `model/hybrid_eval.csv`
    """
    global_path = Path("model/global_eval.csv")
    global_overall_path = Path("model/global_eval_overall.csv")
    hybrid_path = Path("model/hybrid_eval.csv")

    if global_path.exists():
        try:
            df = pd.read_csv(global_path)
        except Exception:
            print("Global evaluation file is empty or unreadable.")
            return
        if df.empty:
            print("Global evaluation file is empty.")
            return
        print("Evaluation results (Global LSTM with district embedding):")
        # Print overall metrics if available
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
        df.to_csv("model/eval_results.csv", index=False)
        print("Evaluation summary saved -> model/eval_results.csv")
        return

    if hybrid_path.exists():
        try:
            df = pd.read_csv(hybrid_path)
        except Exception:
            print("Hybrid evaluation file is empty or unreadable.")
            return
        if df.empty:
            print("Hybrid evaluation file is empty.")
            return
        print("Evaluation results (Hybrid):")
        for _, row in df.iterrows():
            district = row.get("district", "?")
            mae = row.get("Hybrid_MAE", float("nan"))
            rmse = row.get("Hybrid_RMSE", float("nan"))
            print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        df.to_csv("model/eval_results.csv", index=False)
        print("Evaluation summary saved -> model/eval_results.csv")
        return

    print("No evaluation files found. Run training first.")


def run_evaluate():
    """Wrapper to match main.py expectation."""
    evaluate_all()


if __name__ == "__main__":
    evaluate_all()
