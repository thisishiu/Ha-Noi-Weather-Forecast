import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(district_csv, model_dir="model/lstm", lookback=24, epochs=10):
    df = pd.read_csv(district_csv)
    features = df.select_dtypes(float).values

    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i+lookback])
        y.append(features[i+lookback])
    X, y = np.array(X), np.array(y)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = LSTMModel(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"{Path(district_csv).stem}: Epoch {ep+1}/{epochs} - Loss={total/len(dl):.6f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/{Path(district_csv).stem}_lstm.pt")
    print(f"✅ LSTM saved for {Path(district_csv).stem}")

if __name__ == "__main__":
    for f in Path("data/processed").glob("*.csv"):
        train_lstm(f)
