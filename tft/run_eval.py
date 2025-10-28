import argparse
from pathlib import Path
from evaluate import evaluate_all, compute_overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFT model")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory containing global_tft.pt")
    parser.add_argument("--overall-only", action="store_true", help="Only print/save overall metrics")
    parser.add_argument("--recompute", action="store_true", help="Recompute overall metrics from model+test set")
    args = parser.parse_args()

    if args.recompute:
        compute_overall(args.model_dir)
    else:
        evaluate_all(args.model_dir, overall_only=args.overall_only)


if __name__ == "__main__":
    main()
