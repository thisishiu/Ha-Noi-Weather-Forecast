# =========================================================
# PREDICT SERVER — Conformal Prediction + Daily Confidence
# =========================================================
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
data_path <- "N:\\workspace\\DSR301m\\git_project\\data\\weather_date_2.csv"

data <- read_csv(data_path, show_col_types = FALSE)
if ("Ten_Huyen" %in% names(data)) data <- rename(data, district = Ten_Huyen)
data <- data %>% mutate(datetime = ymd_hms(datetime), district = as.character(district))

# ----------------------- MODEL SETUP -----------------------
model_12h_path <- "/model/tft_12h.onnx"
model_7d_path  <- "/model/tft_7d.onnx"

session_12h <- onnxruntime$InferenceSession(model_12h_path)
session_7d  <- onnxruntime$InferenceSession(model_7d_path)

features <- c(
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "apparent_temperature", "pressure_msl", "surface_pressure",
  "precipitation", "rain", "snowfall", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
  "shortwave_radiation"
)

# ----------------------- HELPERS -----------------------

# 48h gần nhất cho model 12h
get_last_48h <- function(df, district_name) {
  df %>%
    filter(district == district_name) %>%
    arrange(desc(datetime)) %>%
    slice_head(n = 48) %>%
    arrange(datetime)
}

# 30 ngày gần nhất cho model 7d
get_last_30d <- function(df, district_name) {
  df %>%
    filter(district == district_name) %>%
    mutate(date = as.Date(datetime)) %>%
    group_by(date) %>%
    summarise(across(all_of(features), \(x) mean(x, na.rm = TRUE))) %>%
    arrange(desc(date)) %>%
    slice_head(n = 30) %>%
    arrange(date)
}

# ----------------------- CONFORMAL PREDICTION FUNCTION -----------------------
predict_model_conformal <- function(df, district_name, model_session, feature_name,
                                   horizon, mode = c("12h", "7d"), alpha = 0.1) {
  mode <- match.arg(mode)

  hist_data <- if (mode == "12h") get_last_48h(df, district_name) else get_last_30d(df, district_name)
  if (nrow(hist_data) < ifelse(mode == "12h", 48, 30))
    stop("⚠️ Không đủ dữ liệu để dự đoán")

  X <- array(as.matrix(hist_data[, features]), dim = c(1L, nrow(hist_data), length(features)))
  X_input <- np$array(X, dtype = "float32")

  district_idx <- match(district_name, unique(df$district)) - 1L
  district_input <- np$array(as.integer(district_idx), dtype = "int64")
  district_input <- np$expand_dims(district_input, axis = 0L)

  out <- model_session$run(list("y_pred"), dict(X = X_input, district_idx = district_input))
  preds_all <- py_to_r(out[[1]])
  preds_feature <- preds_all[1, , match(feature_name, features)]

  # --- CONFORMAL INTERVAL (ước lượng theo residual quá khứ) ---
  past_values <- tail(as.numeric(unlist(hist_data[, feature_name])), 48)
  residuals <- abs(diff(past_values))
  q <- quantile(residuals, probs = 1 - alpha, na.rm = TRUE)

  lower <- preds_feature - q
  upper <- preds_feature + q

  # --- Confidence riêng cho từng điểm ---
  local_var <- zoo::rollapply(past_values, width = 6, FUN = sd, fill = NA, align = "right")
  norm_var <- (local_var - min(local_var, na.rm = TRUE)) /
              (max(local_var, na.rm = TRUE) - min(local_var, na.rm = TRUE) + 1e-6)
  confidence_each <- round(100 - norm_var[seq_len(length(preds_feature))] * 40, 1)
  confidence_each[is.na(confidence_each)] <- mean(confidence_each, na.rm = TRUE)
  confidence_each <- pmax(pmin(confidence_each, 99), 50)

  list(values = preds_feature, lower = lower, upper = upper, confidence_each = confidence_each)
}

# ----------------------- SHINY SERVER -----------------------
predict_server <- function(input, output, session) {

  # ----------------------- 12H FORECAST (Conformal) -----------------------
  output$predict_plot_12h <- renderPlot({
    req(input$predict_selected_region, input$predict_selected_feature)

    pred_obj <- tryCatch({
      predict_model_conformal(data, input$predict_selected_region, session_12h,
                              input$predict_selected_feature, 12, "12h")
    }, error = function(e) {
      showNotification(paste("❌ Lỗi 12h:", e$message), type = "error")
      return(NULL)
    })
    req(pred_obj)

    preds <- pred_obj$values
    lower <- pred_obj$lower
    upper <- pred_obj$upper
    confidence_each <- pred_obj$confidence_each

    start_time <- Sys.time()
    time_seq <- seq(from = start_time + hours(1), by = "hour", length.out = 12)
    df_pred <- data.frame(Time = time_seq, Value = preds, Lower = lower, Upper = upper, Conf = confidence_each)

    ggplot(df_pred, aes(x = Time, y = Value)) +
      geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "skyblue", alpha = 0.3) +
      geom_line(linewidth = 1.3, color = "#e74c3c") +
      geom_point(size = 3, color = "#c0392b") +
      geom_text(aes(label = paste0(round(Value, 2), "\n(", round(Conf, 1), "%)")),
                vjust = -1, size = 3.5, color = "#444") +
      theme_minimal(base_size = 14) +
      labs(
        title = paste("12-Hour Forecast with Conformal Interval —", input$predict_selected_feature),
        x = "Time", y = input$predict_selected_feature
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
        axis.text.x = element_text(angle = 30, hjust = 1)
      )
  })

  # ----------------------- 7-DAY FORECAST (Confidence per day) -----------------------
  output$predict_boxes_7d <- renderUI({
    req(input$predict_selected_region, input$predict_selected_feature)

    pred_obj <- tryCatch({
      predict_model_conformal(data, input$predict_selected_region, session_7d,
                              input$predict_selected_feature, 7, "7d")
    }, error = function(e) {
      showNotification(paste("❌ Lỗi 7d:", e$message), type = "error")
      return(NULL)
    })
    req(pred_obj)

    preds <- pred_obj$values
    confs <- round(pred_obj$confidence_each[1:7], 1)

    today <- Sys.Date()
    days <- seq(today + 1, by = "day", length.out = 7)
    df_pred <- data.frame(Day = weekdays(days), Date = days, Value = round(preds, 2), Conf = confs)

    fluidRow(
      style = "display:flex; justify-content:center; flex-wrap:wrap; gap:15px;",
      lapply(1:7, function(i) {
        tags$div(
          style = paste0(
            "width: 180px; background: linear-gradient(135deg, #4facfe, #00f2fe); color:white;",
            "padding:15px; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.2);"
          ),
          tags$h4(df_pred$Day[i], style = "margin-bottom:8px; font-weight:bold;"),
          tags$p(format(df_pred$Date[i], "%d %b"), style = "font-size:13px; margin-bottom:8px;"),
          tags$div(
            style = "font-size:30px; font-weight:700; margin-bottom:5px;",
            paste0(df_pred$Value[i])
          ),
          tags$p(input$predict_selected_feature, style = "font-size:13px; margin-bottom:4px;"),
          tags$div(
            style = "font-size:13px; color:#f0f0f0;",
            paste0("Confidence: ", df_pred$Conf[i], "%")
          )
        )
      })
    )
  })
}
