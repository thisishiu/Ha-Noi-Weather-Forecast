library(sf)
library(dplyr)
library(stringi)
library(data.table)

source("dashboard/setting.R")
system("python dashboard/get_data.py")

# geometry df 
if (!file.exists(geojson_path)) {
    cat(paste0("[data loader] No geojson file at ", geojson_path ," config in setting.R ", "\n"))
    q()
} else {
    df_geo <- st_read(geojson_path)
    df_geo <- df_geo[df_geo$Ten_Tinh == "Hà Nội", ]
}


# main df 
df <- fread(data_path)
df$datetime <- as.POSIXct(df$datetime)
# df$datetime <- as.Date(df$datetime)
df$district_fix <- stri_trans_general(df$district, "Latin-ASCII")
# df$lon <- NULL
# df$lat <- NULL
df <- arrange(df, datetime, district)

# collumn use to ana 
stats_col <- c("temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation", "rain")


# average each district 
df_mean <- df %>%
        group_by(district_fix, district) %>%
        summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = 'drop')

df_mean <- df_geo %>%
    left_join(df_mean, by = c("Ten_Huyen" = "district_fix"))
df_mean <- df_mean %>% select(geometry, district, all_of(stats_col))

# df of all district 
df_hanoi <- df %>%
    select(datetime, hour, day, month, year, all_of(stats_col)) %>%
    group_by(datetime, hour, day, month, year) %>%
    summarise(
        across(all_of(stats_col), \(x) mean(x, na.rm = TRUE))
    ) %>%
    mutate(
        district = "Hà Nội"
    )
df_hanoi <- as.data.table(df_hanoi)


# add df_hanoi to main df 
df <- rbind(df, df_hanoi, fill=TRUE)
df <- as.data.table(df)

# list of district in Ha Noi (include Ha Noi) 
district_list <- unique(df$district)


