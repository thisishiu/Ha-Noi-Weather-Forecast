import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN  # Đảm bảo đã cài đặt keras-tcn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- 0. Cấu hình thư mục lưu trữ và Tối ưu hóa cho CPU ---
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True) # Tự động tạo thư mục results/

# Cấu hình TensorFlow để chỉ sử dụng CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- 1. Định nghĩa hàm metric R² tùy chỉnh ---
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0))) # Axis=0 để tính mean cho từng feature
    # Tránh chia cho 0 nếu ss_tot quá nhỏ hoặc bằng 0
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# --- 2. Đọc file CSV ---
file_path = 'data.csv'  # Thay thế bằng đường dẫn đến file CSV của bạn
try:
    df = pd.read_csv(file_path)
    print(f"Đã đọc thành công file: {file_path}")
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")
    print("5 dòng đầu tiên của dữ liệu:")
    print(df.head())
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'. Vui lòng kiểm tra lại.")
    exit() # Thoát chương trình nếu không tìm thấy file

# --- 3. Bỏ cột 'district' ---
if 'district' in df.columns:
    df = df.drop(columns=['district'])
    print("Đã bỏ cột 'district'.")
else:
    print("Cột 'district' không tồn tại trong dữ liệu.")

# --- 4. Xử lý cột 'datetime' ---
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 7)
df['dayofyear_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365)
df['dayofyear_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
df = df.drop(columns=['datetime'])
print("Đã xử lý cột 'datetime' bằng sin/cos encoding.")
print(f"Kích thước dữ liệu sau khi xử lý datetime: {df.shape}")

# --- 5. Xác định các cột đầu vào (features) và đầu ra (targets) ---
features_columns = [col for col in df.columns if col not in ['snowfall', 'lat', 'lon']]
target_columns = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'pressure_msl', 'surface_pressure',
    'precipitation', 'rain', 'cloud_cover', 'wind_speed_10m',
    'wind_direction_10m', 'wind_gusts_10m',
    'shortwave_radiation', 'et0_fao_evapotranspiration'
]

missing_targets = [col for col in target_columns if col not in df.columns]
if missing_targets:
    print(f"Cảnh báo: Các cột mục tiêu sau không tìm thấy trong dữ liệu: {missing_targets}")
    target_columns = [col for col in target_columns if col not in missing_targets]
    if not target_columns:
        print("Lỗi: Không còn cột mục tiêu nào hợp lệ. Vui lòng kiểm tra lại cấu trúc dữ liệu.")
        exit()

df_features = df[features_columns]
df_targets = df[target_columns]

print(f"Số lượng đặc trưng đầu vào: {len(features_columns)}")
print(f"Số lượng đặc trưng đầu ra (mục tiêu): {len(target_columns)}")

# --- 6. Chuẩn hóa dữ liệu (MinMaxScaler) ---
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(df_features)

scaler_targets = MinMaxScaler(feature_range=(0, 1))
scaled_targets = scaler_targets.fit_transform(df_targets)

print("Đã chuẩn hóa dữ liệu đầu vào và đầu ra bằng MinMaxScaler.")
print(f"Dữ liệu đầu vào đã chuẩn hóa có dạng: {scaled_features.shape}")
print(f"Dữ liệu đầu ra đã chuẩn hóa có dạng: {scaled_targets.shape}")

# --- 7. Tạo chuỗi time steps ---
def create_sequences(features_data, target_data, timesteps, target_timesteps=1):
    X, y = [], []
    for i in range(len(features_data) - max(timesteps) - target_timesteps + 1):
        x_sequence = []
        for t_step in timesteps:
            x_sequence.append(features_data[i + max(timesteps) - t_step])
        X.append(np.array(x_sequence))
        y.append(target_data[i + max(timesteps) + target_timesteps -1])

    return np.array(X), np.array(y)

TIME_STEPS = [1, 2, 3, 4, 5, 24]
MAX_TIMESTEP = max(TIME_STEPS)

print(f"Sử dụng các bước thời gian: {TIME_STEPS}")

train_size = int(len(scaled_features) * 0.8)
features_train_raw, features_test_raw = scaled_features[:train_size], scaled_features[train_size:]
targets_train_raw, targets_test_raw = scaled_targets[:train_size], scaled_targets[train_size:]

X_train, y_train = create_sequences(features_train_raw, targets_train_raw, TIME_STEPS)
X_test, y_test = create_sequences(features_test_raw, targets_test_raw, TIME_STEPS)

print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước y_train: {y_train.shape}")
print(f"Kích thước X_test: {X_test.shape}")
print(f"Kích thước y_test: {y_test.shape}")

num_features = scaled_features.shape[1]
sequence_length = len(TIME_STEPS)
num_targets = scaled_targets.shape[1]

# --- 8. Huấn luyện mô hình Temporal Convolutional Network (TCN) ---
def build_tcn_model(input_shape, output_units, num_filters=64, kernel_size=2, dilations=[1, 2, 4, 8, 16], dropout_rate=0.2):
    input_layer = Input(shape=input_shape)
    tcn_output = TCN(nb_filters=num_filters,
                     kernel_size=kernel_size,
                     dilations=dilations,
                     padding='causal',
                     use_skip_connections=True,
                     dropout_rate=dropout_rate,
                     return_sequences=False,
                     name='tcn_layer')(input_layer)
    output_layer = Dense(output_units, activation='linear', name='output_layer')(tcn_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

input_shape = (sequence_length, num_features)
output_units = num_targets

model = build_tcn_model(input_shape, output_units,
                        num_filters=128,
                        kernel_size=3,
                        dilations=[1, 2, 4, 8, 16, 32],
                        dropout_rate=0.2)

# Biên dịch mô hình với các metrics yêu cầu
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae',
                       tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                       tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                       r2_score])

model.summary()

BATCH_SIZE = 1024
EPOCHS = 30

print(f"\nBắt đầu huấn luyện mô hình TCN với batch_size={BATCH_SIZE}, epochs={EPOCHS}...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.1,
                    verbose=1)

print("Huấn luyện mô hình đã hoàn tất.")

# --- 9. Lưu toàn bộ history.history ra file .csv ---
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(RESULTS_DIR, 'training_history.csv')
history_df.to_csv(history_csv_path, index=False)
print(f"✅ Đã lưu training history tại: {history_csv_path}")

# --- 10. Vẽ và lưu các biểu đồ metrics ---
metrics_to_plot = {
    'Loss': ('loss', 'val_loss', 'loss.png'),
    'MAE': ('mae', 'val_mae', 'mae.png'),
    'RMSE': ('rmse', 'val_rmse', 'rmse.png'),
    'MAPE': ('mape', 'val_mape', 'mape.png'),
    'R² Score': ('r2_score', 'val_r2_score', 'r2.png') # Tên metric tùy chỉnh có thể khác trong history.history
}

# Kiểm tra tên metric thực tế từ history.history
# Tên metric tùy chỉnh 'r2_score' có thể xuất hiện là 'r2_score' hoặc 'r2_score_fn' trong history.history
# Tên metric trong history.history sẽ là tên được đặt trong compile, hoặc tên hàm nếu không đặt.
# Ta sẽ duyệt qua history.history.keys() để tìm tên chính xác
actual_r2_metric_name = None
for key in history.history.keys():
    if 'r2_score' in key:
        actual_r2_metric_name = key
        break

if actual_r2_metric_name:
    print(f"Tìm thấy tên metric R² trong history: {actual_r2_metric_name}")
    metrics_to_plot['R² Score'] = (actual_r2_metric_name, 'val_' + actual_r2_metric_name, 'r2.png')
else:
    print("Cảnh báo: Không tìm thấy metric 'r2_score' trong history.history. Biểu đồ R² có thể không được vẽ.")
    # Xóa R2 khỏi danh sách vẽ nếu không tìm thấy
    if 'R² Score' in metrics_to_plot:
        del metrics_to_plot['R² Score']


for metric_name, (train_key, val_key, file_name) in metrics_to_plot.items():
    if train_key in history.history and val_key in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[train_key], label=f'Training {metric_name}')
        plt.plot(history.history[val_key], label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, file_name))
        plt.close() # Đóng biểu đồ để tránh chồng chéo khi vẽ nhiều ảnh
    else:
        print(f"Cảnh báo: Không tìm thấy dữ liệu cho {metric_name} ({train_key}, {val_key}) trong lịch sử huấn luyện.")

print(f"✅ Đã lưu các biểu đồ Loss, MAE, RMSE, MAPE, R² tại thư mục {RESULTS_DIR}/")

# --- 11. Đánh giá mô hình trên tập kiểm tra ---
print("\nĐánh giá mô hình trên tập kiểm tra...")
test_results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"Kết quả Test: {dict(zip(model.metrics_names, test_results))}")


# --- 12. Dự đoán và vẽ biểu đồ so sánh Predicted vs Actual cho 3 cột đầu tiên ---
print("\nTiến hành dự đoán trên một vài mẫu từ tập kiểm tra và hiển thị kết quả...")
num_samples_to_predict = 100 # Lấy nhiều mẫu hơn để biểu đồ rõ ràng hơn
predictions_scaled = model.predict(X_test[:num_samples_to_predict], batch_size=BATCH_SIZE)
actuals_scaled = y_test[:num_samples_to_predict]

# Đảo ngược chuẩn hóa để xem giá trị thực tế
predictions_actual = scaler_targets.inverse_transform(predictions_scaled)
actuals_actual = scaler_targets.inverse_transform(actuals_scaled)

plt.figure(figsize=(15, 8))
for i in range(min(3, num_targets)): # Vẽ cho 3 cột mục tiêu đầu tiên
    plt.subplot(3, 1, i + 1)
    plt.plot(actuals_actual[:, i], label='Actual', alpha=0.7)
    plt.plot(predictions_actual[:, i], label='Predicted', alpha=0.7)
    plt.title(f'{target_columns[i]}: Predicted vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
predicted_vs_actual_path = os.path.join(RESULTS_DIR, 'predicted_vs_actual.png')
plt.savefig(predicted_vs_actual_path)
plt.close() # Đóng biểu đồ
print(f"✅ Đã lưu biểu đồ Predicted vs Actual tại: {predicted_vs_actual_path}")


# --- 13. Lưu mô hình đã huấn luyện ---
model_save_path = os.path.join(RESULTS_DIR, 'tcn_weather_prediction_model_2.keras')
model.save(model_save_path)
print(f"Mô hình đã được lưu tại: {model_save_path}")

# --- 14. Thông báo hoàn tất ---
print("\n--- HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ ---")
print(f"✅ Đã lưu training history tại: {history_csv_path}")
print(f"✅ Đã lưu các biểu đồ Loss, MAE, RMSE, MAPE, R² và Predicted vs Actual tại thư mục {RESULTS_DIR}/")