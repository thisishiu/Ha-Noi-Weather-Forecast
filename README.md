# Dự báo thời tiết theo quận (Hà Nội) bằng Global LSTM (Python/PyTorch)

README này mô tả đúng những gì repo hiện đang triển khai: một pipeline Python dự báo chuỗi thời gian đa biến cho từng quận của Hà Nội bằng một mô hình LSTM toàn cục (global) với embedding quận và embedding tọa độ (Fourier) từ lat/lon.

---

## Tổng quan

- Dữ liệu đầu vào: `data/hanoi_weather.csv` chứa các cột tối thiểu `datetime`, `district`, `lat`, `lon` và các đặc trưng thời tiết.
- Tiền xử lý: chuẩn hóa tên quận, sắp xếp theo thời gian, chọn đặc trưng, tách train/dev/test theo từng quận, scale MinMax theo train.
- Huấn luyện: một mô hình Global LSTM dùng embedding theo quận + đặc trưng địa lý Fourier từ `lat/lon`; huấn luyện trên dữ liệu gộp (từng quận vẫn tạo cửa sổ thời gian riêng).
- Đánh giá: xuất MAE, RMSE theo từng quận trên bộ test kết hợp.
- Trực quan: vẽ loss toàn cục và biểu đồ Pred vs Actual cho tất cả đặc trưng của một quận.

---

## Thư mục chính

- `data/`
  - `hanoi_weather.csv`: dữ liệu thô (đồng bộ cột như phần “Đặc trưng”).
  - `splits/`: dữ liệu sau tách và scale: `train.csv`, `dev.csv`, `test.csv`.
- `src/`
  - `preprocess.py`: tiền xử lý + tách train/dev/test (kết hợp giữa các quận).
  - `train_global.py`: định nghĩa dataset, kiến trúc Global LSTM, train và lưu mô hình.
  - `evaluate.py`: đọc `model/global_eval.csv` (hoặc `model/hybrid_eval.csv` nếu có) và in kết quả.
  - `visualize.py`: vẽ `model/global_loss.csv` và biểu đồ Pred vs Actual theo quận.
  - `main.py`: chạy tuần tự 4 bước: preprocess → train → evaluate → visualize.
- `model/`
  - Tạo trong quá trình chạy: `global_lstm.pt`, `global_config.json`, `global_loss.csv`, `global_eval.csv`, và hình trong `model/figures/`.

---

## Cách chạy nhanh

1) Cài môi trường (Python ≥ 3.10). Khuyến nghị:

```
pip install -r requirements.txt
```

Hoặc cài thủ công (tương đương): `pip install numpy pandas scikit-learn matplotlib torch`.

2) Chuẩn bị dữ liệu: đặt file `data/hanoi_weather.csv` với các cột:

- Bắt buộc: `datetime` (parse được datetime), `district` (tên quận), `lat`, `lon`.
- Đặc trưng được dùng (nếu có sẽ giữ lại):
  - `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`,
  - `surface_pressure`, `precipitation`, `cloud_cover`, `wind_speed_10m`.

3) Chạy pipeline:

```
python src/preprocess.py      # tiền xử lý + tách + scale
python src/train_global.py    # huấn luyện Global LSTM
python src/evaluate.py        # in kết quả MAE/RMSE theo quận
python src/visualize.py       # lưu hình loss + pred_vs_actual
```

Hoặc chạy tất cả một lần:

```
python src/main.py
```

4) Thiết lập tuỳ chọn hình vẽ (tuỳ chọn):

```
# Vẽ Pred vs Actual (tất cả đặc trưng) cho một quận cụ thể
set PLOT_DISTRICT=Thanh_Tri  # Windows PowerShell dùng $env:PLOT_DISTRICT="Thanh_Tri"
python src/visualize.py
```

---

## Đầu ra chính

- `model/global_lstm.pt`: trọng số mô hình Global LSTM.
- `model/global_config.json`: cấu hình huấn luyện, danh sách đặc trưng, mapping quận → index.
- `model/global_loss.csv`: lịch sử loss train/val; hình: `model/figures/global_loss.png`.
- `model/global_eval.csv`: MAE, RMSE theo quận; tổng hợp: `model/eval_results.csv`.
- `model/figures/global_pred_vs_actual_all_<DISTRICT>.png`: ví dụ: `model/figures/global_pred_vs_actual_all_Thanh_Tri.png`.

---

## Chi tiết kỹ thuật ngắn gọn

- Cửa sổ đầu vào: `lookback=24` (mặc định), dự báo đa biến 1 bước cho tất cả đặc trưng.
- Embedding: `nn.Embedding` cho quận; embedding tọa độ lấy từ `lat/lon` qua Fourier features.
- Tối ưu: Adam, ReduceLROnPlateau, early stopping theo loss validation.
- Tách dữ liệu: 70/15/15 theo thời gian riêng cho từng quận, sau đó gộp lại thành `train/dev/test.csv` chung.
- Scale: MinMaxScaler fit trên train, áp dụng cho dev/test.
- Thiết bị: tự động dùng GPU nếu có (`torch.cuda.is_available()`).

Thay đổi siêu tham số: sửa trực tiếp các tham số mặc định trong `src/train_global.py` hàm `train_global(...)` nếu cần.

---
