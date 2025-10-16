import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import openmeteo_requests
from retry_requests import retry
import requests_cache
import os

# --- Setup cache & retry ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
openmeteo = openmeteo_requests.Client(session=retry_session)
url = "https://api.open-meteo.com/v1/forecast"

# --- File CSV đầu ra ---
output_file = "data//weather_hanoi_latest.csv"

# --- DataFrame các quận/huyện ---
# Giả sử bạn có 
hanoi = pd.read_csv("data//centroid_hanoi.csv")
tub_huyen = hanoi[["Ten_Huyen", "lon", "lat"]]
# --- Lấy thời gian hiện tại (theo giờ VN) ---
now = datetime.now(timezone.utc) + timedelta(hours=7)
end_date = now.strftime("%Y-%m-%d")

# Nếu file cũ đã tồn tại → đọc để xác định start_date mới
if os.path.exists(output_file):
    old_data = pd.read_csv(output_file, parse_dates=["date"])
    last_time = old_data["date"].max()
    last_time = last_time.tz_localize("Asia/Bangkok") if last_time.tzinfo is None else last_time
    start_date = last_time.strftime("%Y-%m-%d")
    print(f"🔁 Phát hiện file cũ. Lấy dữ liệu từ {start_date} → {end_date}")
else:
    old_data = pd.DataFrame()
    start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"🆕 Lần đầu chạy. Lấy dữ liệu từ {start_date} → {end_date}")

all_data = []

for _, row in tub_huyen.iterrows():
    name = row["Ten_Huyen"]
    lat = row["lat"]
    lon = row["lon"]

    print(f"📡 Gọi API cho {name} ({lat:.4f}, {lon:.4f})...")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "rain",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Bangkok",
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        hourly = response.Hourly()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert("Asia/Bangkok")

        df = pd.DataFrame({
            "Ten_Huyen": name,
            "date": times,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(3).ValuesAsNumpy(),
            "rain": hourly.Variables(4).ValuesAsNumpy(),
        })

        # Cắt đến thời điểm hiện tại
        df = df[df["date"] <= now]
        all_data.append(df)
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Lỗi với {name}: {e}")

# --- Gộp tất cả quận/huyện ---
new_data = pd.concat(all_data, ignore_index=True)
new_data = new_data.sort_values(by=["date"])

# --- Nếu có dữ liệu cũ thì append phần mới ---
if not old_data.empty:
    combined = pd.concat([old_data, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ten_Huyen", "date"])  # tránh trùng
else:
    combined = new_data

combined = combined.sort_values(by=["date"])
combined.to_csv(output_file, index=False)

print(f"✅ Hoàn tất cập nhật! Đã lưu {output_file}")
