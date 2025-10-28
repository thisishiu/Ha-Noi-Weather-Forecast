# libs
library(shiny)
library(shinydashboard)
library(reticulate)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

# -------------------------
# 0. Python / ONNX setup
# -------------------------
# sửa path python nếu cần
use_python("C:/Users/PC/AppData/Local/Programs/Python/Python39/python.exe", required = TRUE)

onnx <- import("onnx")
onnxruntime <- import("onnxruntime")
np <- import("numpy", convert = FALSE)

# -------------------------
# 1. Load CSV data (an toàn hơn)
# -------------------------
# Nếu file lớn và có vấn đề, dùng fill = TRUE khi cần; giữ read_csv nhưng bảo toàn cột datetime
data_path <- "N:\\workspace\\DSR301m\\git_project\\model\\weather_date_2.csv"

# đọc file, nếu có lỗi định dạng, dùng read_lines -> xử lý hoặc dùng tryCatch
data <- tryCatch(
  read_csv(data_path, show_col_types = FALSE),
  error = function(e) {
    # fallback: dùng readr với guess_max lớn hơn hoặc read.csv base với fill = TRUE
    message("Warning: read_csv failed, trying read.csv with fill = TRUE")
    df <- read.csv(data_path, stringsAsFactors = FALSE, fill = TRUE)
    return(df)
  }
)

# chuẩn hóa tên cột: nếu cột tên là Ten_Huyen / district, chuẩn hóa sang 'district'
if ("Ten_Huyen" %in% names(data)) {
  data <- rename(data, district = Ten_Huyen)
}
if (!("datetime" %in% names(data))) {
  stop("File data phải có cột 'datetime'.")
}

# parse datetime, giữ nguyên nếu parse thất bại
data <- data %>%
  mutate(
    datetime = ymd_hms(datetime, quiet = TRUE),
    district = as.character(district)
  )

# -------------------------
# 2. Hàm lấy 48h gần nhất
# -------------------------
get_last_48h <- function(df, district_name) {
  df %>%
    filter(district == district_name) %>%
    filter(!is.na(datetime)) %>%
    arrange(desc(datetime)) %>%
    slice_head(n = 48) %>%
    arrange(datetime)
}

# -------------------------
# 3. Load ONNX model (InferenceSession)
# -------------------------
model_path <- "N:/workspace/DSR301m/git_project/model/tft_12h.onnx"
if (!file.exists(model_path)) {
  warning("Model ONNX not found at: ", model_path, ". predict_district() will error if called.")
  session <- NULL
} else {
  session <- onnxruntime$InferenceSession(model_path)
}

# -------------------------
# 4. predict_district: trả về vector numeric (length 12)
# -------------------------
predict_district <- function(df, district_name) {
  if (is.null(session)) {
    stop("ONNX session chưa được load. Không thể predict.")
  }

  features <- c(
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "pressure_msl", "surface_pressure",
    "precipitation", "rain", "snowfall", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation"
  )

  data_48h <- get_last_48h(df, district_name)
  if (nrow(data_48h) < 1) {
    stop("Không tìm được dữ liệu cho district: ", district_name)
  }

  missing_cols <- setdiff(features, names(data_48h))
  if (length(missing_cols) > 0) {
    stop(paste("Thiếu các cột:", paste(missing_cols, collapse = ", ")))
  }

  # đảm bảo numeric
  data_48h <- data_48h %>% mutate(across(all_of(features), ~ as.numeric(.x)))

  # X có shape [1, 48, n_features]
  X_array <- array(as.matrix(data_48h[, features]), dim = c(1L, nrow(data_48h), length(features)))
  X_input <- np$array(X_array, dtype = "float32")

  # district index: match với danh sách district trong df (0-index nếu model cần)
  district_list <- unique(df$district)
  district_idx <- match(district_name, district_list) - 1L
  if (is.na(district_idx)) district_idx <- 0L  # fallback

  # tạo array đúng rank
  district_input <- np$array(as.integer(district_idx), dtype = "int64")
  district_input <- np$expand_dims(district_input, axis = 0L)  # shape [1]

  # run session (tên input tùy model, ở đây dùng 'X' và 'district_idx' như ví dụ)
  out <- session$run(NULL, dict(X = X_input, district_idx = district_input))

  # out[[1]] có thể là numpy array; chuyển về R
  r_out <- tryCatch(py_to_r(out[[1]]), error = function(e) out[[1]])
  # flatten thành vector numeric
  r_vec <- as.numeric(r_out)

  # nếu model trả về nhiều hơn 12 giá trị, lấy 12 FIRST; nếu ít hơn, lỗi
  if (length(r_vec) < 12) {
    stop("predict_district: output length <", 12, " (found ", length(r_vec), ")")
  }
  return(r_vec[1:12])
}

# -------------------------
# 5. Shiny server (module-like)
# -------------------------
predict_server <- function(input, output, session) {

  # reactive lấy predict vector từ model
  predict_probs <- reactive({
    req(input$predict_selected_region)
    # gọi hàm, trả về numeric vector length 12
    vec12 <- predict_district(data, input$predict_selected_region)
    vec12
  })

  # render UI cho forecast boxes (12 ô)
  output$predict_forecast_ui <- renderUI({
    req(predict_probs())
    probs <- predict_probs()
    rain_threshold <- 0.5

    div(
      style = "display: flex; overflow-x: auto; gap: 20px; padding: 20px; scroll-behavior: smooth;",
      lapply(seq_along(probs), function(i) {
        prob <- probs[i]
        is_rain <- prob > rain_threshold
        icon_class <- ifelse(is_rain, "fas fa-cloud-rain", "fas fa-sun")
        text_color <- ifelse(is_rain, "#0c2461", "#e67e22")
        bg_gradient <- ifelse(is_rain,
                              "linear-gradient(145deg, #d6eaff, #74b9ff)",
                              "linear-gradient(145deg, #fff8e1, #ffeaa7)")

        weather_text <- ifelse(is_rain,
                               paste0("Rainy — ", round(prob * 100), "%"),
                               paste0("Sunny — ", round((1 - prob) * 100), "%"))

        tags$div(
          style = paste0(
            "min-width: 150px; max-width: 150px;",
            "background:", bg_gradient, ";",
            "border-radius: 20px;",
            "box-shadow: 0 4px 10px rgba(0,0,0,0.15);",
            "padding: 20px; text-align:center;",
            "transition: all 0.3s ease-in-out; flex-shrink: 0;"
          ),
          class = "predict-weather-box",
          tags$i(class = icon_class, style = paste0("font-size: 45px; color:", text_color, ";")),
          tags$h4(weather_text, style = paste0("color:", text_color, "; font-weight: 600; margin-top: 10px;")),
          tags$p(paste0("+", i, "h"), style = "color: gray; font-size: 14px; margin-top: 5px;")
        )
      })
    )
  })

  # renderPlot: Actual (48h) vs Predicted (12h) using ggplot2 (bền & đẹp)
  output$predict_plot <- renderPlot({
  req(input$predict_selected_region)

  # ---- 1️⃣ Lọc dữ liệu thực tế 48h ----
  weather_filtered <- data %>%
    filter(district == input$predict_selected_region) %>%
    arrange(desc(datetime))

  if (nrow(weather_filtered) < 48) {
    showNotification("⚠️ Dữ liệu chưa đủ 48h cho district này!", type = "warning")
    return(NULL)
  }

  weather_48h <- weather_filtered %>%
    slice_head(n = 48) %>%
    arrange(datetime)

  time_actual <- weather_48h$datetime
  actual <- weather_48h$rain

  # ---- 2️⃣ Lấy dự đoán 12h ----
  predict_values <- tryCatch({
    as.numeric(predict_district(data, input$predict_selected_region))
  }, error = function(e) {
    showNotification("❌ Lỗi khi dự đoán, vui lòng kiểm tra model ONNX!", type = "error")
    return(NULL)
  })
  req(predict_values)

  # ---- 3️⃣ Tạo thời gian cho dự đoán ----
  last_time <- max(time_actual)
  time_predict <- seq(from = last_time + 3600, by = 3600, length.out = 12)

  # ---- 4️⃣ Tạo dataframe tổng ----
  df_actual <- data.frame(Time = time_actual, Value = actual, Type = "Actual (48h)")
  df_predict <- data.frame(Time = time_predict, Value = predict_values, Type = "Predicted (12h)")

  df_total <- rbind(df_actual, df_predict)

  # ---- 5️⃣ Vẽ bằng ggplot2 ----
  library(ggplot2)

  ggplot(df_total, aes(x = Time, y = Value, color = Type, linetype = Type)) +
    geom_line(linewidth = 1.3) +
    geom_point(size = 2) +
    scale_color_manual(values = c("Actual (48h)" = "#1f77b4", "Predicted (12h)" = "#de2d26")) +
    scale_linetype_manual(values = c("Actual (48h)" = "solid", "Predicted (12h)" = "dashed")) +
    theme_minimal(base_size = 14) +
    labs(
      title = paste("Actual (48h) vs Predicted (12h) —", input$predict_selected_region),
      x = "Time",
      y = "Rain Probability",
      color = "Legend",
      linetype = "Legend"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, color = "#2b2b2b"),
      axis.text.x = element_text(angle = 30, hjust = 1, color = "#555555"),
      axis.text.y = element_text(color = "#555555"),
      legend.position = "top",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray85", linetype = "dotted")
    )
})

} # end predict_server
