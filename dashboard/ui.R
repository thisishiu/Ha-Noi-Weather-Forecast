library(leaflet)
library(shiny)
library(shinydashboard)
# library(modules)

source("dashboard/setting.R")
source("dashboard/modules/home_ui.R")


# Hà Nội Weather Forecast Dashboard

ui <- dashboardPage(
    dashboardHeader(title = "Hà Nội Weather Forecast", titleWidth = bar_width),
    dashboardSidebar(
        width = bar_width,
        sidebarMenu(
            menuItem("Home", tabName = "home", icon = icon("home")),
            menuItem("Statistic", tabName = "statistic", icon = icon("chart-bar")),
            menuItem("History", tabName = "history", icon = icon("clock-rotate-left")),
            menuItem("Predict", tabName = "predict", icon = icon("cloud"))
        )
    ),
    dashboardBody(
        tabItems(
            homeTab("home"),
            tabItem(tabName = "statistic",
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

