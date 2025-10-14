library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

home_server <- function(input, output, session) {
    selected_district_id <- reactiveVal(NULL)

    observeEvent(input$weather_map_shape_click, {
        click <- input$weather_map_shape_click
        highlight_id <- click$id
        selected_district_id(highlight_id)
        
        leafletProxy("weather_map", data = df_mean) %>%
            clearShapes() %>%
            addPolygons(
            layerId = ~district,
            fillColor = ~ifelse(district == highlight_id, "#FF6666", "#99CCFF"),
            color = "#131313",
            weight = 1,
            fillOpacity = 0.7
            )
    })

    observeEvent(input$home_reset_btn, {
        selected_district_id(NULL)
        
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
    filtered_data <- reactive({
        if (is.null(selected_district_id())) {
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
                filter(district == selected_district_id()) %>%
                select(district, all_of(stats_col))
        }
    })

    output$weather_map <- renderLeaflet({
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

    output$district_box <- renderValueBox({
        data <- filtered_data()
        div(
            id = "district_box",
            valueBox(
                data$district[1],
                subtitle = "",
                icon = icon("map-marker-alt"),
                color = "light-blue"
            )
        )
    })

    output$rain_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$rain[1], 1), " mm"), 
            paste0("Rain"), 
            icon = icon("cloud-rain"),
            color = "aqua"
        )
    })

    # T2M
    output$temp_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$temperature_2m[1], 1), " °C"), 
            paste0("Temperature"), 
            icon = icon("thermometer-half"),
            color = "red"
        )
    })

    # Humidity
    output$humidity_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$relative_humidity_2m[1], 1), " %"), 
            paste0("Humidity"), 
            icon = icon("droplet"),
            color = "blue"
        )
    })

    # Wind Speed
    output$wind_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$wind_speed_10m[1], 1), " km/h"), 
            paste0("Wind Speed"), 
            icon = icon("wind"),
            color = "teal"
        )
    })
}