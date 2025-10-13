library(leaflet)
library(shiny)
library(shinydashboard)

# Hà Nội Weather Forecast Dashboard
bar_width <- 300


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
                tags$div(
                    h1("Welcome to Hà Nội Weather Forecast Dashboard"),
                    style = "text-align: center; margin-bottom: 25px;" 
                ),
                fluidRow(
                    column(
                        width = 3,
                        fluidRow(
                            valueBoxOutput("wind_box", width = 12),
                        ),
                        fluidRow(
                            valueBoxOutput("humidity_box", width = 12), 
                        ),
                        fluidRow(
                            valueBoxOutput("temp_box", width = 12)
                        )
                    ),
                    column(
                        width = 9,
                        div(
                            style = "display: flex; justify-content: center;",
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

