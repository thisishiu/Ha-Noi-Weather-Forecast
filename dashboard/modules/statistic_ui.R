library(shiny)
library(shinydashboard)
library(shinyWidgets)

source("dashboard/setting.R")

statisticTab <- function(tabName) {
    return(
        tabItem(tabName = tabName,
            tags$div(
                h1("Weather Statistics"),
            ),
            fluidRow(
                column(
                    width = panel_width,
                    # calendar
                    fluidRow(
                        airDatepickerInput(
                            inputId = "statistic_calendar",
                            # value = Sys.Date(),
                            inline = TRUE, 
                            dateFormat = "dd-mm-yyyy",
                            language = "en",
                        )
                    ),
                    # district chosing
                    fluidRow(
                        pickerInput(
                            inputId = "statistic_district_select",
                            label = NULL,
                            choices = district_list,
                            multiple = TRUE,
                            selected = c("Hà Nội"),
                            options = list(
                                `actions-box` = TRUE,
                                `live-search` = TRUE, 
                                `selected-text-format` = "count > 3"
                            )
                        )
                    )
                ),
                column(
                    width = 12 - panel_width,
                    fluidRow(
                        plotOutput("statistic_plot_A", height = "400px")
                    ),
                    fluidRow(
                        column(
                            width = 6,
                            plotOutput("statistic_info_B", height = "400px")
                        ),
                        column(
                            width = 6,
                            plotOutput("statistic_info_C", height = "400px")
                        )
                    )
                ),
            )
        )
    )
}