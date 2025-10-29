library(leaflet)
library(shiny)
library(shinydashboard)

historyTab <- function(tabName){
    return(
        tabItem(tabName = tabName,
        fluidRow(
                box(title = "Filter Options", width = 12,
                    dateRangeInput("date_range", "Select Date Range:",
                                   start = "2022-01-01", end = "2022-12-31"),
                    selectInput("district", "Select District:", choices = NULL)
                )
              ),
              fluidRow(
                box(title = "Weather Data", width = 12,
                    DT::dataTableOutput("weather_table"))
              )
    )
    )
}