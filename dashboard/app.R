library(shiny)
library(shinydashboard)

source("dashboard/ui.R")
source("dashboard/server.R")

runApp(
    shinyApp(ui, server),
    port = 8080, 
    host = "127.0.0.1"
)
