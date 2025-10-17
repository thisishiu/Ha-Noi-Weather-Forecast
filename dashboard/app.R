library(shiny)
library(shinydashboard)

addResourcePath("dashboard", "dashboard")

source("dashboard/server.R")
source("dashboard/ui.R")


runApp(
    shinyApp(ui, server),
    port = 8080, 
    host = "127.0.0.1"
)
