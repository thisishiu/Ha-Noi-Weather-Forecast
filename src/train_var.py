import pandas as pd
from statsmodels.tsa.api import VAR
from pathlib import Path
import joblib

def train_var(district_csv, model_dir="model/var"):
    df = pd.read_csv(district_csv, parse_dates=["datetime"])
    df = df.select_dtypes(float)
    model = VAR(df)
    result = model.fit(maxlags=3)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(result, f"{model_dir}/{Path(district_csv).stem}_var.pkl")
    print(f"✅ VAR saved for {Path(district_csv).stem}")

if __name__ == "__main__":
    for file in Path("data/processed").glob("*.csv"):
        train_var(file)
