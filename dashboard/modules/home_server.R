library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

home_server <- function(input, output, session) {

    # --- Biến reactive để lưu khu vực đã click ---
    home_selected_district_id <- reactiveVal(NULL)

    #  CLICK: highlight + lưu khu vực 
    observeEvent(input$home_weather_map_shape_click, {
        home_click <- input$home_weather_map_shape_click
        home_highlight_id <- home_click$id
        home_selected_district_id(home_highlight_id)

        leafletProxy("home_weather_map", data = df_mean) %>%
            clearShapes() %>%
            addPolygons(
                layerId = ~district,
                fillColor = ~ifelse(district == home_highlight_id, "#FF6666", "#99CCFF"),
                color = "#131313",
                weight = 1,
                fillOpacity = 0.7
            )
    })

    #  RESET MAP 
    observeEvent(input$home_reset_btn, {
        home_selected_district_id(NULL)
        leafletProxy("home_weather_map", data = df_mean) %>%
            clearShapes() %>%
            addPolygons(
                layerId = ~district,
                fillColor = "#99CCFF",
                color = "#131313",
                weight = 1,
                fillOpacity = 0.7
            )
    })

    #  NÚT PREDICT (HIỆN SAU KHI CLICK) 
    output$home_predict_btn <- renderUI({
        req(home_selected_district_id())
        tags$div(
            style = "text-align:center; margin-top:15px;",
            actionButton(
                inputId = "home_go_predict",
                label = paste("🔮 Predict for", home_selected_district_id()),
                class = "btn btn-success",
                style = "font-weight:bold; border-radius:12px; width:80%;"
            )
        )
    })

    # Khi nhấn nút Predict → chuyển sang tab Predict
    observeEvent(input$home_go_predict, {
        district_clicked <- home_selected_district_id()
        if (!is.null(district_clicked)) {
            updateSelectInput(session, "predict_selected_region", selected = district_clicked)
            updateTabItems(session, "tabs", "predict")
            showNotification(paste("🔍 Predicting:", district_clicked), type = "message")
        }
    })

    #  DỮ LIỆU CỦA KHU VỰC 
    home_filtered_data <- reactive({
        if (is.null(home_selected_district_id())) {
            df_mean %>%
                mutate(
                    across(
                        .cols = -c(geometry, district),
                        .fns = ~ mean(.x, na.rm = TRUE)
                    ),
                    district = "Hà Nội"
                )
        } else {
            df_mean %>%
                filter(district == home_selected_district_id()) %>%
                select(district, all_of(stats_col))
        }
    })

    #  HIỂN THỊ MAP 
    output$home_weather_map <- renderLeaflet({
        leaflet(df_mean) %>%
            addTiles() %>%
            addPolygons(
                layerId = ~district,
                fillColor = "#99CCFF",
                color = "#131313",
                weight = 1,
                fillOpacity = 0.7,
                popup = ~district
            )
    })

    #  VALUE BOXES 
    output$home_district_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            home_data$district[1],
            subtitle = "",
            icon = icon("map-marker-alt"),
            color = "light-blue"
        )
    })

    output$home_rain_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$rain[1], 1), " mm"),
            "Rain",
            icon = icon("cloud-rain"),
            color = "aqua"
        )
    })

    output$home_temp_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$temperature_2m[1], 1), " °C"),
            "Temperature",
            icon = icon("thermometer-half"),
            color = "red"
        )
    })

    output$home_humidity_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$relative_humidity_2m[1], 1), " %"),
            "Humidity",
            icon = icon("droplet"),
            color = "blue"
        )
    })

    output$home_wind_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$wind_speed_10m[1], 1), " km/h"),
            "Wind Speed",
            icon = icon("wind"),
            color = "teal"
        )
    })
}
