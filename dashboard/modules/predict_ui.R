library(shiny)
library(shinydashboard)
library(fontawesome)

# --- Danh sách các loại thời tiết ---
weather_options <- list(
    "Temperature (°C)" = "temperature_2m",
    "Humidity (%)" = "relative_humidity_2m",
    "Dew Point (°C)" = "dew_point_2m",
    "Apparent Temperature (°C)" = "apparent_temperature",
    "Pressure MSL (hPa)" = "pressure_msl",
    "Surface Pressure (hPa)" = "surface_pressure",
    "Precipitation (mm)" = "precipitation",
    "Rain (mm)" = "rain",
    "Snowfall (mm)" = "snowfall",
    "Cloud Cover (%)" = "cloud_cover",
    "Wind Speed (m/s)" = "wind_speed_10m",
    "Wind Direction (°)" = "wind_direction_10m",
    "Wind Gusts (m/s)" = "wind_gusts_10m",
    "Shortwave Radiation (W/m²)" = "shortwave_radiation"
)

predictTab <- function(tabName) {
  	tabItem(
    	tabName = tabName,

    	tags$div(
      		h1("Weather Forecast Dashboard",
         	style = "text-align:center; font-weight:bold; color:#2b2b2b; margin-bottom:25px;"
      		)
    	),

		# --- Chọn loại thời tiết & khu vực ---
		fluidRow(
		style = "display:flex; justify-content:center; gap:20px; margin-bottom:20px;",
		column(
			width = 3,
			selectInput("predict_selected_feature", "Select weather feature:", weather_options)
		),
		column(
			width = 3,
			selectInput("predict_selected_region", "Select region:", district_list)
		)
		),

		# --- Biểu đồ dự đoán 12h ---
		tags$div(
		tags$div(
			style = "background:white; border-radius:20px; padding:20px; 
					box-shadow:0 4px 15px rgba(0,0,0,0.1); margin-bottom:40px;",
			plotOutput("predict_plot_12h", height = "400px")
		)
		),

		# --- Thẻ dự báo 7 ngày ---
		tags$div(
		h3("7-Day Forecast",
			style = "text-align:center; font-weight:700; margin-top:25px;"
		),
		uiOutput("predict_boxes_7d")
		)
	)
}
