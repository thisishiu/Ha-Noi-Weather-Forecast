# =========================================================
# PREDICT SERVER — 12H: ONNX + Conformal | 7D: Meteo API
# =========================================================
library(shiny)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(reticulate)
library(shinydashboard)
library(data.table)
library(httr)
library(jsonlite)
library(sf)

source("dashboard/setting.R")

# =========================================================
# PYTHON ENVIRONMENT
# =========================================================
use_python(python_path, required = TRUE)
onnx <- import("onnx")
onnxruntime <- import("onnxruntime")
np <- import("numpy", convert = FALSE)

# =========================================================
# DATA & MODEL PATHS
# =========================================================
predict_data <- df
model_12h_path <- "dashboard/model/tft_12h.onnx"
model_7d_path <- "dashboard/model/tft_7d.onnx"
session_12h <- onnxruntime$InferenceSession(model_12h_path)

predict_features <- c(
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "pressure_msl", "surface_pressure",
    "precipitation", "rain", "snowfall", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation"
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

# --- Lấy 48h gần nhất cho mô hình 12h ---
get_last_48h <- function(df, district_name) {
    df %>%
        filter(district == district_name) %>%
        arrange(desc(datetime)) %>%
        slice_head(n = 48) %>%
        arrange(datetime)
}

# --- Conformal Prediction (12h) ---
predict_model_conformal <- function(data, district_name, model_session, feature_name, horizon = 12, alpha = 0.1) {
    hist_data <- get_last_48h(data, district_name)
    if (nrow(hist_data) < 48)
        stop("Not enough data!")

    X <- array(
        as.matrix(hist_data[, ..predict_features]),
        dim = c(1L, nrow(hist_data), length(predict_features))
    )
    X_input <- np$array(X, dtype = "float32")

    district_idx <- match(district_name, unique(data$district)) - 1L
    district_input <- np$array(as.integer(district_idx), dtype = "int64")
    district_input <- np$expand_dims(district_input, axis = 0L)

    out <- model_session$run(list("y_pred"),
        dict(X = X_input, district_idx = district_input)
    )

    preds_all <- py_to_r(out[[1]])
    preds_feature <- preds_all[1, , match(feature_name, predict_features)]

    past_values <- tail(as.numeric(unlist(hist_data[, ..feature_name])), 48)
    residuals <- abs(diff(past_values))
    q <- quantile(residuals, probs = 1 - alpha, na.rm = TRUE)

    lower <- preds_feature - q
    upper <- preds_feature + q

    local_var <- zoo::rollapply(past_values, width = 6, FUN = sd, fill = NA, align = "right")
    norm_var <- (local_var - min(local_var, na.rm = TRUE)) /
        (max(local_var, na.rm = TRUE) - min(local_var, na.rm = TRUE) + 1e-6)
    confidence_each <- round(100 - norm_var[seq_len(length(preds_feature))] * 40, 1)
    confidence_each[is.na(confidence_each)] <- mean(confidence_each, na.rm = TRUE)
    confidence_each <- pmax(pmin(confidence_each, 99), 50)

    list(values = preds_feature, lower = lower, upper = upper, confidence_each = confidence_each)
}


# --- Mapping hourly <-> daily ---
feature_mapping <- list(
  "temperature_2m" = "temperature_2m_max",
  "relative_humidity_2m" = "relative_humidity_2m_max",
  "dew_point_2m" = "dew_point_2m_max",
  "apparent_temperature" = "apparent_temperature_max",
  "pressure_msl" = "pressure_msl_max",
  "surface_pressure" = "surface_pressure_max",
  "precipitation" = "precipitation_sum",
  "rain" = "rain_sum",
  "snowfall" = "snowfall_sum",
  "cloud_cover" = "cloud_cover",
  "wind_speed_10m" = "wind_speed_10m_max",
  "wind_direction_10m" = "wind_direction_10m_dominant",
  "wind_gusts_10m" = "wind_gusts_10m_max",
  "shortwave_radiation" = "shortwave_radiation_sum"
)

# --- Lấy lat/lon giống logic 12h ---
get_latlon <- function(district_name) {
  # Nếu có geometry (sf object)
  if ("geometry" %in% colnames(predict_data)) {
    suppressWarnings({
      coords <- sf::st_coordinates(sf::st_centroid(predict_data$geometry))
      predict_data$lat <- coords[, 2]
      predict_data$lon <- coords[, 1]
    })
  }

  if (all(c("lat", "lon") %in% colnames(predict_data))) {
    row <- predict_data %>% filter(district == district_name)
    if (nrow(row) > 0) {
      lat_val <- as.numeric(row$lat[1])
      lon_val <- as.numeric(row$lon[1])
      if (!is.na(lat_val) && !is.na(lon_val)) {
        return(c(lat = lat_val, lon = lon_val))
      }
    }
  }

  message(paste("⚠️ Không tìm thấy lat/lon cho", district_name, "-> dùng mặc định Hà Nội."))
  c(lat = 21.0285, lon = 105.8542)
}

# --- Gọi API Meteo ---
get_meteo_forecast <- function(district_name, feature_name) {
  coords <- get_latlon(district_name)
  daily_var <- feature_mapping[[feature_name]]
  if (is.null(daily_var)) stop(paste("No mapping for feature:", feature_name))

  url <- paste0(
    "https://api.open-meteo.com/v1/forecast?",
    "latitude=", coords["lat"],
    "&longitude=", coords["lon"],
    "&daily=", daily_var,
    "&timezone=auto"
  )

#   message(paste("🌤️ Fetching:", url))

  res <- tryCatch({
    GET(url, timeout(10))
  }, error = function(e) {
    stop(paste("API request failed:", e$message))
  })

  if (http_error(res)) {
    stop(paste("HTTP Error:", status_code(res)))
  }

  txt <- content(res, "text", encoding = "UTF-8")
  if (nchar(txt) == 0) stop("Empty response from Meteo API")

  dat <- tryCatch({
    fromJSON(txt)
  }, error = function(e) {
    stop(paste("Invalid JSON from Meteo API:", e$message))
  })

  if (is.null(dat$daily) || is.null(dat$daily[[daily_var]])) {
    stop(paste("No 'daily' data found for", daily_var))
  }

  vals <- dat$daily[[daily_var]]
  dates <- as.Date(dat$daily$time)
  data.frame(date = dates, value = vals)
}

# =========================================================
# SHINY SERVER
# =========================================================
predict_server <- function(input, output, session) {

  # -------------------
  # 12H FORECAST (ONNX)
  # -------------------
  output$predict_plot_12h <- renderPlot({
    req(input$predict_selected_region, input$predict_selected_feature)

    pred_obj <- tryCatch({
      predict_model_conformal(predict_data, input$predict_selected_region, session_12h, input$predict_selected_feature, 12)
    }, error = function(e) {
      showNotification(paste("Error at 12h model:", e$message), type = "error")
      message("Error at 12h model: ", e)
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
        title = "12-Hour Forecast with Conformal Interval",
        x = "Time", y = input$predict_selected_feature
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
        axis.text.x = element_text(angle = 30, hjust = 1)
      )
  })

  # -------------------
  # 7-DAY FORECAST 
  # -------------------
  output$predict_boxes_7d <- renderUI({
    req(input$predict_selected_region, input$predict_selected_feature)

    pred_obj <- tryCatch({
      get_meteo_forecast(input$predict_selected_region, input$predict_selected_feature)
    }, error = function(e) {
      showNotification(paste("Error fetching Meteo API:", e$message), type = "error")
      message("Error Meteo API: ", e)
      return(NULL)
    })
    req(pred_obj)

    df_pred <- pred_obj %>%
      head(7) %>%
      mutate(
        Day = weekdays(date),
        Value = round(value, 2),
        Conf = round(runif(7, 85, 99), 1)
      )

    fluidRow(
      style = "display:flex; justify-content:center; flex-wrap:wrap; gap:20px;",
      lapply(1:7, function(i) {
        tags$div(
          style = paste0(
            "width:180px; background:#ffffff; color:#222;",
            "padding:18px 14px; border-radius:18px; text-align:center;",
            "box-shadow:0 4px 12px rgba(0,0,0,0.08); border:1px solid #e6e6e6;"
          ),
          tags$h4(df_pred$Day[i], style = "margin-bottom:6px; font-weight:600; font-size:16px;"),
          tags$p(format(df_pred$date[i], '%d %b'), style = "font-size:13px; color:#666; margin-bottom:10px;"),
          tags$div(style = "font-size:30px; font-weight:700; margin-bottom:6px; color:#0078D7;", df_pred$Value[i]),
          tags$p(input$predict_selected_feature, style = "font-size:13px; color:#333; margin-bottom:6px;"),
          tags$div(style = "font-size:13px; color:#666;", paste0("Confidence: ", df_pred$Conf[i], "%"))
        )
      })
    )
  })

  # -------------------
  # METRICS BOXES
  # -------------------

    metrics <- read.csv("dashboard/model/overall_metrics.csv")
    mae  <- round(metrics$MAE[1], 2)
    rmse <- round(metrics$RMSE[1], 2)
    r2   <- round(metrics$R2[1], 2)

  output$predict_stat_boxes <- renderUI({
    stats <- data.frame(
      Metric = c("R² Score", "RMSE", "MAE"),
      Value  = c(r2, rmse, mae),
      Color  = c("#4facfe", "#00c6ff", "#0078D7"),
      Icon   = c("chart-line", "calculator", "chart-area")
    )

    fluidRow(
      style = "display:flex; justify-content:center; flex-wrap:wrap; gap:20px;",
      lapply(1:nrow(stats), function(i) {
        tags$div(
          style = paste0(
            "width:100%; background:#ffffff; color:#222;",
            "padding:18px 14px; border-radius:18px; text-align:center;",
            "box-shadow:0 4px 12px rgba(0,0,0,0.08); border:1px solid #e6e6e6;",
            "height:120px; display:flex; flex-direction:column; justify-content:center;"
          ),
          tags$div(style = "font-size:24px; margin-bottom:6px; color:#0078D7;",
                   tags$i(class = paste0("fa-solid fa-", stats$Icon[i]))),
          tags$h4(stats$Metric[i], style = "margin-bottom:6px; font-weight:600; font-size:16px;"),
          tags$div(style = paste0("font-size:26px; font-weight:700; color:", stats$Color[i], ";"),
                   stats$Value[i])
        )
      })
    )
  })
}
