import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np

def preprocess(input_path="data/hanoi_weather.csv", output_dir="data/processed"):
    print(f"🚀 Loading {input_path} ...")
    df = pd.read_csv(input_path, parse_dates=["datetime"])
    df = df.dropna(subset=["datetime"]).sort_values(["district", "datetime"]).reset_index(drop=True)

    # === Các cột đặc trưng cần dùng ===
    features = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "surface_pressure", "precipitation",
        "cloud_cover", "wind_speed_10m"
    ]

    # === Bỏ các cột không dùng hoặc lỗi ===
    drop_cols = [
        "snowfall", "shortwave_radiation", "et0_fao_evapotranspiration",
        "lat", "lon"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # === Giữ lại các cột có trong features + meta ===
    keep_cols = ["datetime", "district"] + [f for f in features if f in df.columns]
    df = df[keep_cols].copy()

    # === Chuẩn hóa MinMax cho các cột numeric ===
    scaler = MinMaxScaler()
    numeric_cols = [col for col in features if col in df.columns]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # === Tạo thư mục đầu ra ===
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === Lưu file theo từng quận ===
    for district, group in df.groupby("district"):
        name = district.replace(" ", "_")
        out_file = Path(output_dir) / f"{name}.csv"
        group.to_csv(out_file, index=False)
        print(f"✅ Saved {name}: {len(group)} rows → {out_file}")

    print("✨ Preprocessing complete!")
    return df


if __name__ == "__main__":
    preprocess()

