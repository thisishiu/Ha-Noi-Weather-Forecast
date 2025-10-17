import os
import time
import pytz
import requests_cache
import openmeteo_requests
from retry_requests import retry
from datetime import datetime, timedelta, timezone
import pandas as pd

# --- Setup cache & retry ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
openmeteo = openmeteo_requests.Client(session=retry_session)
url = "https://api.open-meteo.com/v1/forecast"


df = pd.read_csv("data/weather_date_2.csv", parse_dates=['datetime'])
print(min(df['datetime']), max(df['datetime']))
district_list = df[["district", "lat", "lon"]].drop_duplicates()
data_wrap = []

now = datetime.now(timezone.utc) + timedelta(hours=7)
end_date = now.strftime("%Y-%m-%d")
start_time = df['datetime'].max()
start_time = start_time.tz_localize("Asia/Bangkok")
start_date = start_time.strftime("%Y-%m-%d")

# print(now)
# print(start_date)
# print(end_date)

data = {
    "Ten_Huyen": [],
    "datetime": [],
    "temperature_2m": [],
    "relative_humidity_2m": [],
    "wind_speed_10m": [],
    "cloud_cover": [],
    "rain": [],
    "lat": [],
    "lon": [],
}

for _, row in district_list.iterrows():
    name = row["district"]
    lat = row["lat"]
    lon = row["lon"]

    # print(f"get: {name} ({lat:.4f}, {lon:.4f})")

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

        n = len(times)
        data["Ten_Huyen"].extend([name] * n)
        data["datetime"].extend(times)
        data["temperature_2m"].extend(hourly.Variables(0).ValuesAsNumpy())
        data["relative_humidity_2m"].extend(hourly.Variables(1).ValuesAsNumpy())
        data["wind_speed_10m"].extend(hourly.Variables(2).ValuesAsNumpy())
        data["cloud_cover"].extend(hourly.Variables(3).ValuesAsNumpy())
        data["rain"].extend(hourly.Variables(4).ValuesAsNumpy())
        data["lat"].extend([lat] * n)
        data["lon"].extend([lon] * n)

        time.sleep(0.2)  

    except Exception as e:
        print(f"{name}: {e}")

new_data = pd.DataFrame(data)

new_data["datetime"] = pd.to_datetime(new_data["datetime"], utc=True).dt.tz_convert("Asia/Bangkok")
end_time = end_date = now.strftime("%Y-%m-%d %H:00:00")
new_data = new_data[(new_data["datetime"] > start_time) & (new_data["datetime"] <= end_time)]



# # --- Gộp tất cả quận/huyện ---
district_map = {
    "Ba Vi": "Ba Vì",
    "Ba Dinh": "Ba Đình",
    "Chuong My": "Chương Mỹ",
    "Cau Giay": "Cầu Giấy",
    "Gia Lam": "Gia Lâm",
    "Hai Ba Trung": "Hai Bà Trưng",
    "Hoai Duc": "Hoài Đức",
    "Hoan Kiem": "Hoàn Kiếm",
    "Hoang Mai": "Hoàng Mai",
    "Ha Dong": "Hà Đông",
    "Long Bien": "Long Biên",
    "Me Linh": "Mê Linh",
    "My Duc": "Mỹ Đức",
    "Phu Xuyen": "Phú Xuyên",
    "Phuc Tho": "Phúc Thọ",
    "Quoc Oai": "Quốc Oai",
    "Soc Son": "Sóc Sơn",
    "Son Tay": "Sơn Tây",
    "Thanh Oai": "Thanh Oai",
    "Thanh Tri": "Thanh Trì",
    "Thanh Xuan": "Thanh Xuân",
    "Thuong Tin": "Thường Tín",
    "Thach That": "Thạch Thất",
    "Tay Ho": "Tây Hồ",
    "Tu Liem": "Từ Liêm",
    "Dan Phuong": "Đan Phượng",
    "Dong Anh": "Đông Anh",
    "Dong Da": "Đống Đa",
    "Ung Hoa": "Ứng Hòa"
}
new_data["district"] = new_data["Ten_Huyen"].map(district_map).fillna(new_data["Ten_Huyen"])

for col in df.columns:
    if col not in new_data.columns:
        new_data[col] = pd.NA

new_data = new_data[df.columns]
new_data['datetime'] = new_data['datetime'].dt.tz_convert('Asia/Ho_Chi_Minh')
new_data['hour'] = new_data['datetime'].dt.hour
new_data['day'] = new_data['datetime'].dt.date
new_data['month'] = new_data['datetime'].dt.month
new_data['year'] = new_data['datetime'].dt.year
new_data['datetime'] = new_data['datetime'].dt.tz_localize(None)

# print(new_data.head())
# print(new_data[~new_data['district'].isin(district_map.values())])
new_data.to_csv("data/weather_date_2.csv", mode="a", header=False, index=False)