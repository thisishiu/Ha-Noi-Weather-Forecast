library(sf)
library(dplyr)
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
        group_by(district_fix, district) %>%
        summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = 'drop')

df_mean <- df_geo %>%
    left_join(df_mean, by = c("Ten_Huyen" = "district_fix"))

stats_col <- c("temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation", "rain")
df_mean <- df_mean %>% select(geometry, district, all_of(stats_col))