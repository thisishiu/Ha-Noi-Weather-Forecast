# src/main.py
from preprocess import run_preprocess
from train_global import run_train
from evaluate import run_evaluate
from visualize import run_visualize

if __name__ == "__main__":
    # 1) Tiền xử lý & chia dữ liệu
    run_preprocess()

    # 2) Huấn luyện model global LSTM
    run_train()

    # 3) Đánh giá theo từng quận
    run_evaluate()

    # 4) Vẽ biểu đồ kết quả
    run_visualize()
