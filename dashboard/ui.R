library(leaflet)
library(shiny)
library(shinydashboard)
library(ggplot2)
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
                h2("Statistics Section"),
                selectInput( 
                "selectModel", 
                "Select models below:", 
                list("A" = "A", "Choice 1B" = "1B", "Choice 1C" = "1C") 
        ), 
                textOutput("modelChoice"),
                fluidRow(
                style = "display: flex; align-items: stretch;",
                    column(
                        width = 3,
                        fluidRow(
                            valueBox("1,024", "Số người truy cập", icon = icon("users"), color = "aqua", width = 12),
                        ),
                        fluidRow(
                            valueBox("95%", "Tỷ lệ hoàn thành", icon = icon("check-circle"), color = "green", width = 12),
                        ),
                        fluidRow(
                            valueBox("12", "Lỗi phát hiện", icon = icon("bug"), color = "red", width = 12)
                        ),
                    ),
                    column(
                            width = 9,
                            plotOutput("predictPlot")
                    ),
                        ),
                
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
