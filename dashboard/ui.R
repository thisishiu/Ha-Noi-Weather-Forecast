library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

# Hà Nội Weather Forecast Dashboard

ui <- dashboardPage(
    dashboardHeader(title = "Hà Nội Weather Forecast", titleWidth = bar_width),
    dashboardSidebar(
        width = bar_width,
        sidebarMenu(
            menuItem("Home", tabName = "home", icon = icon("home")),
            menuItem("Statistic", tabName = "stats", icon = icon("chart-bar")),
            menuItem("History", tabName = "history", icon = icon("clock-rotate-left")),
            menuItem("Predict", tabName = "predict", icon = icon("cloud"))
        )
    ),
    dashboardBody(
        tabItems(
            tabItem(tabName = "home",
                tags$head(
                    tags$style(HTML("
                    .small-box {
                        border-radius: 15px !important;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        transition: all 0.3s ease;
                    }

                    .small-box:hover {
                        transform: translateY(-3px);
                        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
                    }

                    #district_box .small-box {
                        background: transparent !important;
                        color: #000 !important;
                        box-shadow: none !important; 
                        font-family: 'Arial', 'Segoe UI', 'Tahoma', 'sans-serif';
                    }

                    #district_box .small-box > .inner > h3,
                    #district_box .small-box > .inner > p {
                        color: #000 !important;
                        font-weight: bold;
                        font-family: 'Arial', 'Segoe UI', 'Tahoma', 'sans-serif';
                    }

                    "))
                ),
                tags$div(
                    h1("Welcome to Hà Nội Weather Forecast Dashboard"),
                    style = "text-align: center; margin-bottom: 25px; font-weight: bold; color: #333;"
                ),
                fluidRow(
                    column(
                        width = info_width,
                        fluidRow(
                            valueBoxOutput("district_box", width = 12),
                        ),
                        fluidRow(
                            valueBoxOutput("temp_box", width = 12)
                        ),
                        fluidRow(
                            valueBoxOutput("rain_box", width = 12), 
                        ),
                        fluidRow(
                            valueBoxOutput("humidity_box", width = 12), 
                        ),
                        fluidRow(
                            valueBoxOutput("wind_box", width = 12),
                        ),
                        fluidRow(
                            column(
                                width = 12,
                                div(
                                style = "text-align: center; margin-top: 10px;",
                                actionButton(
                                        inputId = "reset_btn",
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
                            leafletOutput("weather_map", height = "800px")
                        )
                    )
                )
            ),
            tabItem(tabName = "stats",
                h2("Statistics Section")
            ),
            tabItem(tabName = "history",
                h2("Historical Data Section")
            ),
            tabItem(tabName = "predict",
                h2("Weather Prediction Section")
            )
        )
    )
)

