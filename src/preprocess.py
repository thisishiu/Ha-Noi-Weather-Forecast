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
    df = pd.read_csv(input_path, parse_dates=["datetime"])  # relies on system default utf-8-sig
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Normalize district names
    df["district"] = df["district"].astype(str).map(normalize_district)

    # Sort by district then time
    df = df.sort_values(["district", "datetime"]).reset_index(drop=True)

    # Feature selection
    features: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "surface_pressure",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
    ]
    drop_cols = [
        "snowfall",
        "shortwave_radiation",
        "et0_fao_evapotranspiration",
        "lat",
        "lon",
        "pressure_msl",  # redundant with surface_pressure for our pipeline
        "rain",  # not used
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    keep_cols = ["datetime", "district"] + [f for f in features if f in df.columns]
    df = df[keep_cols]

    # Split per district by time, but output combined CSVs
    parts = {"train": [], "dev": [], "test": []}
    for dname, group in df.groupby("district", sort=False):
        n = len(group)
        if n == 0:
            continue
        s_train, s_dev, s_test = split_indices_per_group(n)
        gtrain = group.iloc[s_train]
        gdev = group.iloc[s_dev]
        gtest = group.iloc[s_test]
        parts["train"].append(gtrain)
        parts["dev"].append(gdev)
        parts["test"].append(gtest)
        print(f"{dname}: Train={len(gtrain)}, Dev={len(gdev)}, Test={len(gtest)}")

    train_df = pd.concat(parts["train"], ignore_index=True) if parts["train"] else pd.DataFrame(columns=keep_cols)
    dev_df = pd.concat(parts["dev"], ignore_index=True) if parts["dev"] else pd.DataFrame(columns=keep_cols)
    test_df = pd.concat(parts["test"], ignore_index=True) if parts["test"] else pd.DataFrame(columns=keep_cols)

    # Fit scaler on train only, apply to all splits
    feat_in_df = [f for f in features if f in train_df.columns]
    if feat_in_df and not train_df.empty:
        scaler = MinMaxScaler()
        scaler.fit(train_df[feat_in_df])
        train_df.loc[:, feat_in_df] = scaler.transform(train_df[feat_in_df])
        if not dev_df.empty:
            dev_df.loc[:, feat_in_df] = scaler.transform(dev_df[feat_in_df])
        if not test_df.empty:
            test_df.loc[:, feat_in_df] = scaler.transform(test_df[feat_in_df])

    # Write combined CSVs
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    dev_df.to_csv(out_dir / "dev.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("Preprocessing + combined splitting done!")


def run_preprocess():
    preprocess()


if __name__ == "__main__":
    preprocess()
