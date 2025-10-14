import pandas as pd
from pathlib import Path


def evaluate_all():
    """Print evaluation results for the best available pipeline.

    Priority:
    1) Global model results at `model/global_eval.csv`
    2) Hybrid per-district results at `model/hybrid_eval.csv`
    """
    global_path = Path("model/global_eval.csv")
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
        for _, row in df.iterrows():
            district = row.get("district", "?")
            mae = row.get("MAE", float("nan"))
            rmse = row.get("RMSE", float("nan"))
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
