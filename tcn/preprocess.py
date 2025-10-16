import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import List

from utils import normalize_district


def split_indices_per_group(n: int, train_ratio: float = 0.7, dev_ratio: float = 0.15):
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    return slice(0, train_end), slice(train_end, dev_end), slice(dev_end, None)


def preprocess(input_path: str = "data/hanoi_weather.csv", output_root: str = "data/splits"):
    base_root = Path(__file__).resolve().parents[1]
    in_path = Path(input_path)
    if not in_path.is_absolute():
        in_path = base_root / in_path
    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = base_root / out_root

    df = pd.read_csv(in_path, parse_dates=["datetime"])
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    df["district"] = df["district"].astype(str).map(normalize_district)
    df = df.sort_values(["district", "datetime"]).reset_index(drop=True)

    use_rain = os.environ.get("USE_RAIN", "").strip() in ("1", "true", "True")
    precip_col = "precipitation"
    if use_rain and "rain" in df.columns:
        precip_col = "rain"
        print("Using 'rain' as precipitation feature (USE_RAIN=1)")
    elif "precipitation" in df.columns:
        precip_col = "precipitation"
        if use_rain:
            print("USE_RAIN=1 set but 'rain' column not found; falling back to 'precipitation'.")
    elif "rain" in df.columns:
        precip_col = "rain"
        print("'precipitation' missing; using 'rain' column.")
    else:
        print("Warning: neither 'precipitation' nor 'rain' found. Proceeding without precipitation-like feature.")
        precip_col = None

    features: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_10m",
    ]
    if precip_col is not None:
        features.insert(5, precip_col)

    # Optional: apply log1p to precipitation-like feature before scaling
    if precip_col is not None and os.environ.get("LOG1P_RAIN", "").strip() in ("1", "true", "True"):
        if precip_col in df.columns:
            # Ensure non-negative before log1p
            vals = df[precip_col].astype(float).values
            vals = np.clip(vals, a_min=0.0, a_max=None)
            df.loc[:, precip_col] = np.log1p(vals)
            print(f"Applied log1p to '{precip_col}' (LOG1P_RAIN=1)")

    drop_cols = [
        "snowfall",
        "shortwave_radiation",
        "et0_fao_evapotranspiration",
        "lat",
        "lon",
        "pressure_msl",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
    if precip_col == "rain" and "rain" in drop_cols:
        drop_cols.remove("rain")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    keep_cols = ["datetime", "district"] + [f for f in features if f in df.columns]
    df = df[keep_cols]

    parts = {"train": [], "dev": [], "test": []}
    for dname, group in df.groupby("district", sort=False):
        n = len(group)
        if n == 0:
            continue
        s_train, s_dev, s_test = split_indices_per_group(n)
        parts["train"].append(group.iloc[s_train])
        parts["dev"].append(group.iloc[s_dev])
        parts["test"].append(group.iloc[s_test])
        print(f"{dname}: Train={len(group.iloc[s_train])}, Dev={len(group.iloc[s_dev])}, Test={len(group.iloc[s_test])}")

    train_df = pd.concat(parts["train"], ignore_index=True) if parts["train"] else pd.DataFrame(columns=keep_cols)
    dev_df = pd.concat(parts["dev"], ignore_index=True) if parts["dev"] else pd.DataFrame(columns=keep_cols)
    test_df = pd.concat(parts["test"], ignore_index=True) if parts["test"] else pd.DataFrame(columns=keep_cols)

    feat_in_df = [f for f in features if f in train_df.columns]
    if feat_in_df and not train_df.empty:
        scaler = MinMaxScaler()
        scaler.fit(train_df[feat_in_df])
        train_df.loc[:, feat_in_df] = scaler.transform(train_df[feat_in_df])
        if not dev_df.empty:
            dev_df.loc[:, feat_in_df] = scaler.transform(dev_df[feat_in_df])
        if not test_df.empty:
            test_df.loc[:, feat_in_df] = scaler.transform(test_df[feat_in_df])

    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    dev_df.to_csv(out_dir / "dev.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print("Preprocessing + combined splitting done!")


def run_preprocess():
    preprocess()


if __name__ == "__main__":
    preprocess()
