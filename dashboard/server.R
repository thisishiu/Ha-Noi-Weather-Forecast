library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/dataLoader.R")

server <- function(input, output, session) {
    selected_district_id <- reactiveVal(NULL)

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
        
        print(paste("Clicked:", highlight_id))
    })

    # Can modify this part to show current weather of selected district (base on df_mean) 
    filtered_data <- reactive({
        req(selected_district_id()) 
        
        df_mean %>%
            filter(district == selected_district_id()) %>%
            select(district, all_of(stats_col))
    })

    # T2M
    output$temp_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$temperature_2m[1], 1), " °C"), 
            paste0("Temperature (", data$district[1], ")"), 
            icon = icon("thermometer-half"),
            color = "red"
        )
    })

    # Humidity
    output$humidity_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$relative_humidity_2m[1], 1), " %"), 
            paste0("Humidity (", data$district[1], ")"), 
            icon = icon("droplet"),
            color = "blue"
        )
    })

    # Wind Speed
    output$wind_box <- renderValueBox({
        data <- filtered_data()
        valueBox(
            paste0(round(data$wind_speed_10m[1], 1), " km/h"), 
            paste0("Wind Speed (", data$district[1], ")"), 
            icon = icon("wind"),
            color = "teal" # Màu xanh lam đậm
        )
    })

    output$predictPlot <- renderPlot({
    # Ví dụ: biểu đồ đơn giản
    x <- 1:10
    y <- x^2
    plot(x, y, type = "b", col = "blue",
         main = "Model A",
         xlab = "X", ylab = "Y = X^2")
  })
  output$modelChoice <- renderText({input$selectModel})

  
    # output$temp_box <- renderValueBox({
    #     valueBox(
    #         "32°C", "Nhiệt độ (Hoàng Mai)", icon = icon("thermometer-half"),
    #         color = "red"
    #     )
    # })

    # output$humidity_box <- renderValueBox({
    #     valueBox(
    #         "68%", "Độ ẩm trung bình", icon = icon("droplet"),
    #         color = "blue"
    #     )
    # })
    
    # output$wind_box <- renderValueBox({
    #     valueBox(
    #         "155", "Chất lượng Không khí", icon = icon("smog"),
    #         color = "orange"
    #     )
    # })
    
    # output$weather_map <- renderPlot({
    #     ggplot(data = df_geo) +
        
    # })

}