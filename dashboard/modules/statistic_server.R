library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

statistic_server <- function(input, output, session) {

    statistic_district_select <- reactiveVal(c("Hà Nội"))
    statistic_calendar <- reactiveVal(Sys.Date())
    statistic_df <- reactiveVal(df_hanoi[as.Date(datetime) == Sys.Date()])

    observeEvent(input$statistic_district_select, {
        statistic_district_select(input$statistic_district_select)
    })

    observeEvent(input$statistic_calendar, {
        statistic_calendar(input$statistic_calendar)
    })

    observeEvent({
        input$statistic_district_select
        input$statistic_calendar
    }, {
        statistic_df(df[
            district %in% statistic_district_select() &
            as.Date(datetime) == as.Date(statistic_calendar())
        ])
    })


    output$statistic_plot_A <- renderPlot({
        req(statistic_df())
        print(statistic_df())
        ggplot(statistic_df(),
            aes(x = hour)
        ) +
            geom_col(aes(y = rain * 15, fill = district), alpha = 0.5, position = "dodge") +
            geom_line(aes(y = temperature_2m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = temperature_2m, color = district), size = 2, alpha = 0.7) +
            scale_x_continuous(
                name = "hour",
                limits = c(0, 23),
                breaks = 0:23
            ) +
            scale_y_continuous(
                name = "temperature (°C)",
                # limits = c(15, 40),
                sec.axis = sec_axis(~ . / 15, 
                    name = "Precipitation (mm)",
                    breaks = 0:5
                )
            ) +
            labs(
                title = "Temperature and Rain by hour",
                color = "District",
                fill = "District"
            ) +
            theme_minimal(base_size = 14)
    })


    output$statistic_info_B <- renderPlot({
        req(statistic_df())
        ggplot(statistic_df(),
            aes(x = hour)
        ) +
            geom_line(aes(y = relative_humidity_2m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = relative_humidity_2m, color = district), size = 2, alpha = 0.7) +
            scale_x_continuous(
                name = "hour",
                limits = c(0, 23),
                breaks = 0:23
            ) +
            scale_y_continuous(
                name = "Relative Humidity (%)",
                limits = c(0, 100),
                breaks = seq(0, 100, by = 10)
            ) +
            labs(
                title = "Relative Humidity by hour",
                color = "District"
            ) +
            theme_minimal(base_size = 14)
    })

    output$statistic_info_C <- renderPlot({
        req(statistic_df())
        ggplot(statistic_df(),
            aes(x = hour)
        ) +
            geom_line(aes(y = wind_speed_10m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = wind_speed_10m, color = district), size = 2, alpha = 0.7) +
            scale_x_continuous(
                name = "hour",
                limits = c(0, 23),
                breaks = 0:23
            ) +
            scale_y_continuous(
                name = "Wind Speed (m/s)",
                limits = c(0, 20),
                breaks = seq(0, 20, by = 2)
            ) +
            labs(
                title = "Wind Speed by hour",
                color = "District"
            ) +
            theme_minimal(base_size = 14)
    })
}