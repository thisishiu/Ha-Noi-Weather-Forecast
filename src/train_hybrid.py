import joblib, torch
import numpy as np
import pandas as pd
from pathlib import Path
from train_lstm import LSTMModel

def predict_hybrid(district, data_dir="data/processed", model_dir="model", lookback=24):
    df = pd.read_csv(f"{data_dir}/{district}.csv")
    features = df.select_dtypes(float).values

    var = joblib.load(f"{model_dir}/var/{district}_var.pkl")
    var_pred = var.forecast(features[-lookback:], steps=lookback)

    lstm = LSTMModel(features.shape[1])
    lstm.load_state_dict(torch.load(f"{model_dir}/lstm/{district}_lstm.pt"))
    lstm.eval()

    with torch.no_grad():
        X = torch.tensor([features[-lookback:]], dtype=torch.float32)
        lstm_pred = lstm(X).numpy()

    hybrid_pred = (var_pred[-1] + lstm_pred[-1]) / 2
    print(f"{district}: Hybrid forecast={hybrid_pred}")
    return hybrid_pred
