library(sf)
library(shiny)
library(dplyr)
library(ggplot2)
library(stringi)
library(data.table)

df <- fread("data/weather_date.csv")
df$datetime <- as.POSIXct(df$datetime)
df$district_fix <- stri_trans_general(df$district, "Latin-ASCII")
df$lon <- NULL
df$lat <- NULL

df_geo <- st_read("data/diaphanhuyen.geojson")
df_geo <- df_geo[df_geo$Ten_Tinh == "Hà Nội", ]

df_mean <- df %>%
        group_by(district_fix) %>%
        summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = 'drop')
df_mean <- df_geo %>%
    left_join(df_mean, by = c("Ten_Huyen" = "district_fix"))

ui <- navbarPage(
    title = "Weather Forecast",
    tabPanel("Map",
        fluidPage(
            titlePanel("Average Weather Map of Hanoi Districts"),
            plotOutput("weather_map", height = "600px")
        )
    )
)

server <- function(input, output) {
    output$weather_map <- renderPlot({
        ggplot(data = df_mean) +
            geom_sf(aes(fill = temperature_2m), color = "white") +
            scale_fill_viridis_c(option = "C") +
            labs(title = "Average Temperature in Hanoi Districts",
                 fill = "Temperature (°C)") +
            theme_minimal()
    })
}


shinyApp(ui, server)