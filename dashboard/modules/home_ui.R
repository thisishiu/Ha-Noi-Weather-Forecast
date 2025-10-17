library(leaflet)
library(shiny)
library(shinydashboard)

homeTab <- function(tabName){
    return(
        tabItem(tabName = tabName,
            tags$div(
                h1("Welcome to Hà Nội Weather Forecast Dashboard"),
                style = "text-align: center; margin-bottom: 25px; font-weight: bold; color: #333;"
            ),
            fluidRow(
                column(
                    width = info_width,
                    fluidRow(
                        valueBoxOutput("home_district_box", width = 12),
                    ),
                    fluidRow(
                        valueBoxOutput("home_temp_box", width = 12)
                    ),
                    fluidRow(
                        valueBoxOutput("home_rain_box", width = 12), 
                    ),
                    fluidRow(
                        valueBoxOutput("home_humidity_box", width = 12), 
                    ),
                    fluidRow(
                        valueBoxOutput("home_wind_box", width = 12),
                    ),
                    fluidRow(
                        column(
                            width = 12,
                            div(
                                style = "text-align: center; margin-top: 10px;",
                                actionButton(
                                        inputId = "home_reset_btn",
                                        label = "Reset to Hà Nội",
                                        icon = icon("undo"),
                                        class = "btn btn-primary",
                                        style = "width: 100%; border-radius: 10px;"
                                    )
                                )
                            )
                        )
                ),
                column(
                    width = map_width,
                    div(
                        style = "
                            display: flex; 
                            justify-content: center; 
                            background-color: #f9f9f9; 
                            padding: 15px; 
                            border-radius: 20px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        ",
                        leafletOutput("home_weather_map", height = "800px")
                    )
                )
            )
        )
    )
}