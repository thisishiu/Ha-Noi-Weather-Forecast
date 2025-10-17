library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")
source("dashboard/dataLoader.R")
source("dashboard/modules/home_server.R")
source("dashboard/modules/statistic_server.R")

server <- function(input, output, session) {
    home_server(input, output, session)
    statistic_server(input, output, session)
}