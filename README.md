# Weather Forecast Dashboard — Hybrid VAR + LSTM Time Series Forecasting per District (Hanoi)

## Overview

This project develops a **time-series-based weather forecasting system** for each **district of Hanoi**, using a **hybrid VAR–LSTM model** that combines statistical forecasting and deep learning.  

The interactive dashboard, built with **R + Shiny**, allows users to:
- Forecast temperature, humidity, wind speed, pressure, solar radiation, and precipitation.
- View short-term (6–24 hour) predictions updated in real time through the **Open-Meteo API**.
- Explore forecast trends by district.

---

## Objectives

- Predict multiple meteorological variables simultaneously (multivariate forecasting).  
- Use past 24–48 hours of data to forecast the next 6–24 hours.  
- Integrate real-time API updates.  
- Provide district-level visualization through a Shiny web dashboard.  
- Combine **VAR (Vector AutoRegression)** and **LSTM (Long Short-Term Memory)** for improved accuracy.

---

## Hybrid VAR–LSTM Model

### Concept

The hybrid model leverages **VAR** for linear relationships and **LSTM** for nonlinear temporal dependencies.

```
Input → VAR model → Forecast_VAR
              ↓
        Residuals = Actual – Forecast_VAR
              ↓
        LSTM model → Predict(Residuals)
              ↓
Final Forecast = Forecast_VAR + LSTM(Residuals)
```

| Component | Purpose |
|------------|----------|
| VAR | Captures short-term linear correlations between weather features |
| LSTM | Learns nonlinear residual patterns over time |
| Hybrid Output | Combines both to enhance stability and accuracy |

---

## Project Structure

```
weather-forecast-team/
│
├── app.R
│
├── R/
│   ├── fetch_weather.R
│   ├── preprocess_data.R
│   ├── feature_engineering.R
│   ├── train_model.R
│   ├── train_hybrid_model.R
│   ├── predict_future.R
│   ├── evaluate_model.R
│   └── utils.R
│
├── data/
│   ├── hanoi_weather.csv
│   ├── district_coords.csv
│   ├── processed/
│   └── realtime_cache/
│
├── model/
│   ├── model_VAR_HoanKiem.Rds
│   ├── model_LSTM_HoanKiem.h5
│   ├── model_VAR_BaVi.Rds
│   └── model_LSTM_BaVi.h5
│
├── dashboard/
│   ├── ui.R
│   ├── server.R
│   ├── theme.R
│   ├── modules/
│   └── www/
│
├── reports/
│   ├── EDA_weather_analysis.Rmd
│   ├── Model_Comparison_Report.Rmd
│   └── Forecast_Results.pdf
│
├── scripts/
│   ├── schedule_update.R
│   ├── retrain_model.R
│   └── deploy_app.R
│
├── docs/
│   ├── project_plan.md
│   └── API_reference.md
│
├── logs/
│   ├── api_log.txt
│   └── model_training_log.txt
│
├── requirements.txt
├── renv.lock
├── Dockerfile
└── README.md
```

---

## Workflow

1. **Data Collection** – Retrieve hourly weather data for each district via the Open-Meteo API and store in `data/hanoi_weather.csv`.
2. **Preprocessing** – Clean missing values, normalize numeric variables, and align timestamps.
3. **Model Training**  
   - Train a VAR model for linear dependencies.  
   - Train an LSTM on the residuals (nonlinear corrections).  
   - Combine both forecasts.
4. **Evaluation** – Compute RMSE, MAE, and compare results between VAR, LSTM, and Hybrid.
5. **Visualization** – Display forecasts and performance metrics on the Shiny dashboard.
6. **Realtime Update** – Refresh predictions every 3 hours using the Open-Meteo API.

---

## Technical Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | R |
| Framework | Shiny, shinydashboard |
| Forecasting | vars, keras, tensorflow, prophet |
| Data Processing | dplyr, tidyr, lubridate, jsonlite |
| Visualization | plotly, ggplot2, leaflet |
| Automation | cronR, reactiveTimer |
| Deployment | ShinyApps.io, Docker |

---

## Installation

### 1. Install required packages

```r
install.packages(c(
  "shiny", "plotly", "httr", "jsonlite", "dplyr",
  "vars", "prophet", "keras", "tensorflow",
  "lubridate", "leaflet", "shinydashboard", "bslib"
))
```

For reproducible environments:

```r
renv::restore()
```

---

### 2. Run the application

```r
shiny::runApp("app.R")
```

Then open:

```
http://localhost:3838/
```

---

### 3. Retrain models (optional)

```r
source("scripts/retrain_model.R")
```

---

### 4. Deploy to ShinyApps.io

```r
rsconnect::deployApp('path/to/app')
```

Or use Docker:

```bash
docker build -t weather-forecast .
docker run -p 3838:3838 weather-forecast
```

---

## Evaluation Results

| Model | RMSE (°C) | MAE (°C) | Notes |
|--------|------------|-----------|--------|
| VAR | 1.82 | 1.45 | Captures short-term linear patterns |
| LSTM | 1.25 | 1.05 | Learns nonlinear dynamics |
| Hybrid VAR–LSTM | **1.08** | **0.92** | Best overall accuracy |

---

## Dashboard Features

| Feature | Description |
|----------|--------------|
| Weather Metrics | Forecasts for temperature, humidity, wind, pressure, radiation, and precipitation |
| Forecast Graphs | Actual vs predicted (VAR, LSTM, Hybrid) |
| District Map | Leaflet map displaying average forecasted temperature |
| Realtime Update | Automatically fetches new data every 3 hours |
| Model Comparison | Visual and numerical comparison across models |

---

## Team Members

| Role | Name | Responsibilities |
|------|------|------------------|
| Data Engineer | A | Data collection and preprocessing |
| Model Developer | B | Build and tune VAR–LSTM models |
| Evaluator | C | Model validation and visualization |
| Frontend Developer | D | Shiny UI and dashboard |
| Backend Integrator | E | API integration and automation |
| Project Manager | F | Documentation and deployment |

---

## Future Work

- Extend forecasting horizon to 48–72 hours using Transformer-based architectures (Informer, TFT).  
- Add AQI (Air Quality Index) forecasting based on weather parameters.  
- Introduce spatial modeling across districts using graph-based learning.  
- Apply explainable AI (DALEX, SHAPforxgboost) for interpretability.  
- Implement alert systems and heatmap layers in the dashboard.

---

## License

This project is for educational and research purposes.  
Weather data is provided by the [Open-Meteo API](https://open-meteo.com/).
