library(shiny)
library(shinydashboard)
library(shinyWidgets)

source("dashboard/setting.R")

statisticTab <- function(tabName) {
    return(
        tabItem(tabName = tabName,
        tags$div(id = "statistic-tab",

            tags$div(
                h1("Weather Statistics"),
            ),
            fluidRow(
                column(
                    width = panel_width,
                    # Time range type selector
                    div(
                        style = "margin-bottom: 15px;",
                        radioButtons(
                            inputId = "statistic_time_range_type",
                            label = "Time Range:",
                            choices = c("Day" = "day", "Month" = "month", "Year" = "year", "Custom Range" = "range"),
                            selected = "day"
                        )
                    ),
                    # Single day picker (shown when Day is selected)
                    conditionalPanel(
                        condition = "input.statistic_time_range_type == 'day'",
                        airDatepickerInput(
                            inputId = "statistic_single_date",
                            label = NULL,
                            value = Sys.Date(),
                            inline = TRUE, 
                            dateFormat = "dd-mm-yyyy",
                            language = "en"
                        )
                    ),
                    # Month picker (shown when Month is selected)
                    conditionalPanel(
                        condition = "input.statistic_time_range_type == 'month'",
                        airDatepickerInput(
                            inputId = "statistic_month",
                            label = NULL,
                            value = Sys.Date(),
                            view = "months",
                            minView = "months",
                            dateFormat = "MM yyyy",
                            inline = TRUE,
                            language = "en"
                        )
                    ),
                    # Year picker (shown when Year is selected)
                    conditionalPanel(
                        condition = "input.statistic_time_range_type == 'year'",
                        airDatepickerInput(
                            inputId = "statistic_year",
                            label = NULL,
                            value = Sys.Date(),
                            view = "years",
                            minView = "years",
                            dateFormat = "yyyy",
                            inline = TRUE,
                            language = "en"
                        )
                    ),
                    # Date range picker (shown when Custom Range is selected)
                    conditionalPanel(
                        condition = "input.statistic_time_range_type == 'range'",
                        airDatepickerInput(
                            inputId = "statistic_date_range",
                            label = NULL,
                            value = c(Sys.Date() - 7, Sys.Date()),
                            range = TRUE,
                            inline = TRUE,
                            dateFormat = "dd-mm-yyyy",
                            language = "en"
                        )
                    ),
                    # district choosing
                    div(
                        style = "margin-top: 20px; margin-bottom: 15px;",
                        pickerInput(
                            inputId = "statistic_district_select",
                            label = "Districts:",
                            choices = district_list,
                            multiple = TRUE,
                            selected = c("Hà Nội"),
                            options = list(
                                `actions-box` = TRUE,
                                `live-search` = TRUE, 
                                `selected-text-format` = "count > 3",
                                `dropupAuto` = FALSE,
                                `width` = "100%"
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
    )
}