library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

history_server <- function(input, output, session) {
        weather_data <- reactive({
        df <- read_csv("data/weather_date_2.csv", show_col_types = FALSE)
        df <- df %>%
        mutate(datetime = as_datetime(datetime)) %>%
        arrange(datetime)
        return(df)
    })
        observe({
        df <- weather_data()
        districts <- sort(unique(df$district))
        updateSelectInput(session, "district", choices = districts, selected = districts[1])
        updateSelectInput(session, "plot_district", choices = districts, selected = districts[1])
    })
    
    # --- Lọc dữ liệu cho bảng ---
    filtered_table <- reactive({
        df <- weather_data()
        df %>%
        filter(district == input$district,
                datetime >= input$date_range[1],
                datetime <= input$date_range[2])
    })
    
    output$weather_table <- DT::renderDataTable({
        DT::datatable(filtered_table(), options = list(pageLength = 10))
    })
}