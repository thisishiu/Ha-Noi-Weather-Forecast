library(dplyr)
library(ggplot2)
library(leaflet)
library(shiny)
library(shinydashboard)

source("dashboard/setting.R")

statistic_server <- function(input, output, session) {

    # Reactive dataframe based on selected districts and time range
    statistic_df <- reactive({
        req(input$statistic_district_select)
        
        districts <- input$statistic_district_select
        time_type <- input$statistic_time_range_type
        
        # Filter by time range based on selected type
        filtered_data <- switch(time_type,
            "day" = {
                req(input$statistic_single_date)
                df[district %in% districts & as.Date(datetime) == as.Date(input$statistic_single_date)]
            },
            "month" = {
                req(input$statistic_month)
                selected_month <- format(as.Date(input$statistic_month), "%Y-%m")
                df[district %in% districts & format(as.Date(datetime), "%Y-%m") == selected_month]
            },
            "year" = {
                req(input$statistic_year)
                selected_year <- format(as.Date(input$statistic_year), "%Y")
                df[district %in% districts & format(as.Date(datetime), "%Y") == selected_year]
            },
            "range" = {
                req(input$statistic_date_range)
                if(length(input$statistic_date_range) == 2) {
                    start_date <- as.Date(input$statistic_date_range[1])
                    end_date <- as.Date(input$statistic_date_range[2])
                    df[district %in% districts & as.Date(datetime) >= start_date & as.Date(datetime) <= end_date]
                } else {
                    df[0, ]  # Return empty if range not properly selected
                }
            },
            # Default to current day
            df[district %in% districts & as.Date(datetime) == Sys.Date()]
        )
        
        return(filtered_data)
    })


    output$statistic_plot_A <- renderPlot({
        req(statistic_df())
        data <- statistic_df()
        time_type <- input$statistic_time_range_type
        
        # Prepare data based on time range type
        plot_data <- if(time_type == "day") {
            data
        } else {
            # Aggregate by date for month/year/range views
            data %>%
                mutate(date = as.Date(datetime)) %>%
                group_by(date, district) %>%
                summarise(
                    temperature_2m = mean(temperature_2m, na.rm = TRUE),
                    rain = sum(rain, na.rm = TRUE),
                    .groups = "drop"
                )
        }
        
        # Calculate dynamic scaling factor for rain
        # Goal: scale rain so its max is about 80% of temperature range
        temp_range <- diff(range(plot_data$temperature_2m, na.rm = TRUE))
        rain_max <- max(plot_data$rain, na.rm = TRUE)
        
        # Avoid division by zero
        if(rain_max > 0 && temp_range > 0) {
            rain_scale <- (temp_range * 0.8) / rain_max
        } else {
            rain_scale <- 1
        }
        
        # Create plot with appropriate x-axis
        p <- if(time_type == "day") {
            ggplot(plot_data, aes(x = hour)) +
                scale_x_continuous(
                    name = "Hour",
                    limits = c(0, 23),
                    breaks = 0:23
                )
        } else {
            ggplot(plot_data, aes(x = date)) +
                scale_x_date(name = "Date", date_labels = "%d/%m")
        }
        
        p +
            geom_col(aes(y = rain * rain_scale, fill = district), alpha = 0.5, position = "dodge") +
            geom_line(aes(y = temperature_2m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = temperature_2m, color = district), size = 2, alpha = 0.7) +
            scale_y_continuous(
                name = "Temperature (°C)",
                sec.axis = sec_axis(~ . / rain_scale, name = "Precipitation (mm)")
            ) +
            labs(
                title = paste("Temperature and Rain -", 
                    switch(time_type, 
                        "day" = "Hourly", 
                        "month" = "Daily (Month)", 
                        "year" = "Daily (Year)",
                        "range" = "Daily (Custom Range)"
                    )),
                color = "District",
                fill = "District"
            ) +
            theme_minimal(base_size = 14) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
    })


    output$statistic_info_B <- renderPlot({
        req(statistic_df())
        data <- statistic_df()
        time_type <- input$statistic_time_range_type
        
        # Prepare data based on time range type
        plot_data <- if(time_type == "day") {
            data
        } else {
            data %>%
                mutate(date = as.Date(datetime)) %>%
                group_by(date, district) %>%
                summarise(
                    relative_humidity_2m = mean(relative_humidity_2m, na.rm = TRUE),
                    .groups = "drop"
                )
        }
        
        # Create plot
        p <- if(time_type == "day") {
            ggplot(plot_data, aes(x = hour)) +
                scale_x_continuous(
                    name = "Hour",
                    limits = c(0, 23),
                    breaks = 0:23
                )
        } else {
            ggplot(plot_data, aes(x = date)) +
                scale_x_date(name = "Date", date_labels = "%d/%m")
        }
        
        p +
            geom_line(aes(y = relative_humidity_2m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = relative_humidity_2m, color = district), size = 2, alpha = 0.7) +
            scale_y_continuous(
                name = "Relative Humidity (%)",
                limits = c(0, 100),
                breaks = seq(0, 100, by = 10)
            ) +
            labs(
                title = paste("Relative Humidity -", 
                    switch(time_type, 
                        "day" = "Hourly", 
                        "month" = "Daily", 
                        "year" = "Daily",
                        "range" = "Daily"
                    )),
                color = "District"
            ) +
            theme_minimal(base_size = 14) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
    })

    output$statistic_info_C <- renderPlot({
        req(statistic_df())
        data <- statistic_df()
        time_type <- input$statistic_time_range_type
        
        # Prepare data based on time range type
        plot_data <- if(time_type == "day") {
            data
        } else {
            data %>%
                mutate(date = as.Date(datetime)) %>%
                group_by(date, district) %>%
                summarise(
                    wind_speed_10m = mean(wind_speed_10m, na.rm = TRUE),
                    .groups = "drop"
                )
        }
        
        # Create plot
        p <- if(time_type == "day") {
            ggplot(plot_data, aes(x = hour)) +
                scale_x_continuous(
                    name = "Hour",
                    limits = c(0, 23),
                    breaks = 0:23
                )
        } else {
            ggplot(plot_data, aes(x = date)) +
                scale_x_date(name = "Date", date_labels = "%d/%m")
        }
        
        p +
            geom_line(aes(y = wind_speed_10m, color = district), linewidth = 2, alpha = 0.7) +
            geom_point(aes(y = wind_speed_10m, color = district), size = 2, alpha = 0.7) +
            scale_y_continuous(
                name = "Wind Speed (m/s)",
                limits = c(0, 20),
                breaks = seq(0, 20, by = 2)
            ) +
            labs(
                title = paste("Wind Speed -", 
                    switch(time_type, 
                        "day" = "Hourly", 
                        "month" = "Daily", 
                        "year" = "Daily",
                        "range" = "Daily"
                    )),
                color = "District"
            ) +
            theme_minimal(base_size = 14) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
    })
}