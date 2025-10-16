from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_global_loss(save_path: str | Path | None = None, show: bool = False):
    base_dir = Path(__file__).resolve().parent
    loss_path = base_dir / "model/global_loss.csv"
    if not loss_path.exists():
        raise FileNotFoundError(f"{loss_path} not found")
    df = pd.read_csv(loss_path)
    if df.empty:
        raise ValueError("global_loss.csv is empty")
    if save_path is None:
        save_path = base_dir / "model/figures/global_loss.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Global LSTM Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def run_visualize():
    try:
        plot_global_loss()
    except Exception as e:
        print(f"Skip global loss plot: {e}")


if __name__ == "__main__":
    run_visualize()

