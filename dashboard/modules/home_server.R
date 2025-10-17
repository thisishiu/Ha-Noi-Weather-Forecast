library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

home_server <- function(input, output, session) {
    home_selected_district_id <- reactiveVal(NULL)

    observeEvent(input$home_weather_map_shape_click, {
        home_click <- input$home_weather_map_shape_click
        home_highlight_id <- home_click$id
        home_selected_district_id(home_highlight_id)

        # print(home_highlight_id)
        
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

    observeEvent(input$home_reset_btn, {
        home_selected_district_id(NULL)
        
        leafletProxy("weather_map", data = df_mean) %>%
            clearShapes() %>%
            addPolygons(
            layerId = ~district,
            fillColor = "#99CCFF",
            color = "#131313",
            weight = 1,
            fillOpacity = 0.7
            )
    })

    # Can modify this part to show current weather of selected district (base on df_mean) 
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

    # Map
    output$home_weather_map <- renderLeaflet({
        leaflet(df_mean) %>%
            addTiles() %>%
                addPolygons(
                    layerId = ~district,
                    fillColor = "#99CCFF",
                    color = "#131313",
                    weight = 1,
                    fillOpacity = 0.7
                )
    })

    # District
    output$home_district_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            home_data$district[1],
            subtitle = "",
            icon = icon("map-marker-alt"),
            color = "light-blue"
        )
    })

    # Rain
    output$home_rain_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$rain[1], 1), " mm"), 
            paste0("Rain"), 
            icon = icon("cloud-rain"),
            color = "aqua"
        )
    })

    # T2M
    output$home_temp_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$temperature_2m[1], 1), " °C"), 
            paste0("Temperature"), 
            icon = icon("thermometer-half"),
            color = "red"
        )
    })

    # Humidity
    output$home_humidity_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$relative_humidity_2m[1], 1), " %"), 
            paste0("Humidity"), 
            icon = icon("droplet"),
            color = "blue"
        )
    })

    # Wind Speed
    output$home_wind_box <- renderValueBox({
        home_data <- home_filtered_data()
        valueBox(
            paste0(round(home_data$wind_speed_10m[1], 1), " km/h"), 
            paste0("Wind Speed"), 
            icon = icon("wind"),
            color = "teal"
        )
    })
}