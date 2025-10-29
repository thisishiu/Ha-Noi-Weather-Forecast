# ----------------------- LIBRARIES -----------------------
library(shiny)
library(shinydashboard)
library(reticulate)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

# ----------------------- PYTHON ENV -----------------------
use_python("C:/Users/PC/AppData/Local/Programs/Python/Python39/python.exe", required = TRUE)
onnx <- import("onnx")
onnxruntime <- import("onnxruntime")
np <- import("numpy", convert = FALSE)

# ----------------------- LOAD DATA -----------------------
data_path <- "N:/workspace/DSR301m/git_project/model/weather_date_2.csv"

data <- read_csv(data_path, show_col_types = FALSE)
if ("Ten_Huyen" %in% names(data)) data <- rename(data, district = Ten_Huyen)
data <- data %>% mutate(datetime = ymd_hms(datetime), district = as.character(district))

# ----------------------- MODEL SETUP -----------------------
model_path <- "N:/workspace/DSR301m/git_project/model/tft_12h.onnx"
if (!file.exists(model_path)) stop("❌ Không tìm thấy model: ", model_path)
session_12h <- onnxruntime$InferenceSession(model_path)

# Các feature đầu vào (thứ tự đúng theo model training)
features <- c(
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "apparent_temperature", "pressure_msl", "surface_pressure",
  "precipitation", "rain", "snowfall", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
  "shortwave_radiation"
)

# ----------------------- HELPER FUNCTIONS -----------------------

# Lấy 48h gần nhất
get_last_48h <- function(df, district_name) {
  df %>%
    filter(district == district_name) %>%
    arrange(desc(datetime)) %>%
    slice_head(n = 48) %>%
    arrange(datetime)
}

# Hàm dự đoán 12h cho 1 feature
predict_12h <- function(df, district_name, model_session, feature_name) {
  hist_data <- get_last_48h(df, district_name)
  if (nrow(hist_data) < 48)
    stop("⚠️ Không đủ dữ liệu 48h để dự đoán")

  # Chuẩn bị input cho model
  X <- array(as.matrix(hist_data[, features]), dim = c(1L, 48L, length(features)))
  X_input <- np$array(X, dtype = "float32")

  district_idx <- match(district_name, unique(df$district)) - 1L
  district_input <- np$array(as.integer(district_idx), dtype = "int64")
  district_input <- np$expand_dims(district_input, axis = 0L)

  # Run model (output tensor = "y_pred")
  out <- model_session$run(list("y_pred"), dict(X = X_input, district_idx = district_input))
  preds_all <- py_to_r(out[[1]])  # shape [1, 12, 14]

  # Xác định feature được chọn
  feature_idx <- match(feature_name, features)
  if (is.na(feature_idx))
    stop(paste("❌ Feature", feature_name, "không tồn tại trong model!"))

  # Lấy riêng 12 giá trị dự đoán cho feature đó
  preds_feature <- preds_all[1, , feature_idx]
  as.numeric(preds_feature)
}

# ----------------------- SHINY SERVER -----------------------

predict_server <- function(input, output, session) {

  # Render biểu đồ dự đoán 12h
  output$predict_plot <- renderPlot({
    req(input$predict_selected_region, input$predict_selected_feature)

    preds <- tryCatch({
      predict_12h(
        data,
        input$predict_selected_region,
        session_12h,
        input$predict_selected_feature
      )
    }, error = function(e) {
      showNotification(paste("❌ Lỗi khi dự đoán:", e$message), type = "error")
      return(NULL)
    })
    req(preds)

    # Tạo thời gian cho 12h dự đoán kế tiếp
    start_time <- Sys.time()
    time_seq <- seq(from = start_time + hours(1), by = "hour", length.out = 12)
    df_pred <- data.frame(Time = time_seq, Value = preds)

    # Vẽ biểu đồ
    ggplot(df_pred, aes(x = Time, y = Value)) +
      geom_line(linewidth = 1.4, color = "#e74c3c") +
      geom_point(size = 3, color = "#c0392b") +
      geom_text(aes(label = round(Value, 2)), vjust = -1, color = "#555555", size = 4) +
      theme_minimal(base_size = 14) +
      labs(
        title = paste("12-Hour Forecast —", input$predict_selected_feature, "in", input$predict_selected_region),
        x = "Time",
        y = input$predict_selected_feature
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
        axis.text.x = element_text(angle = 30, hjust = 1),
        axis.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "gray85", linetype = "dotted")
      )
  })
}
