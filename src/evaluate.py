import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_hybrid import predict_hybrid
from pathlib import Path

def evaluate_all():
    results = []
    for file in Path("data/processed").glob("*.csv"):
        district = Path(file).stem
        pred = predict_hybrid(district)
        df = pd.read_csv(file)
        y_true = df.select_dtypes(float).iloc[-1].values
        mae = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        results.append({"district": district, "MAE": mae, "RMSE": rmse})
        print(f"{district}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    pd.DataFrame(results).to_csv("model/eval_results.csv", index=False)
    print("✅ Evaluation done → model/eval_results.csv")

if __name__ == "__main__":
    evaluate_all()
