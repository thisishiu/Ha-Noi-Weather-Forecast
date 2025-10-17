library(sf)
library(dplyr)
library(stringi)
library(data.table)


system("python dashboard/get_data.py")

# geometry df ------------------------------------------------
df_geo <- st_read("data/diaphanhuyen.geojson")
df_geo <- df_geo[df_geo$Ten_Tinh == "Hà Nội", ]
# ------------------------------------------------------------

# main df ----------------------------------------------------
df <- fread("data/weather_date_2.csv")
df$datetime <- as.POSIXct(df$datetime)
df$district_fix <- stri_trans_general(df$district, "Latin-ASCII")
df$lon <- NULL
df$lat <- NULL
df <- arrange(df, datetime, district)
# ------------------------------------------------------------

# collumn use to ana -----------------------------------------
stats_col <- c("temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation", "rain")
# ------------------------------------------------------------

# average each district --------------------------------------
df_mean <- df %>%
        group_by(district_fix, district) %>%
        summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = 'drop')

df_mean <- df_geo %>%
    left_join(df_mean, by = c("Ten_Huyen" = "district_fix"))
df_mean <- df_mean %>% select(geometry, district, all_of(stats_col))
# ------------------------------------------------------------


# df of all district -----------------------------------------
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
# ------------------------------------------------------------

# add df_hanoi to main df ------------------------------------
df <- rbind(df, df_hanoi, fill=TRUE)
df <- as.data.table(df)
# ------------------------------------------------------------


# list of district in Ha Noi (include Ha Noi) ----------------
district_list <- unique(df$district)
# ------------------------------------------------------------

# repeat {    
#     source("dataLoader.R", local = TRUE)
    
#     cat("Update at:", Sys.time(), "\n")

#     now <- Sys.time()
#     next_hour <- as.POSIXct(format(now, "%Y-%m-%d %H:00:00")) + 3600
#     wait_time <- as.numeric(difftime(next_hour, now, units = "secs"))

#     Sys.sleep(wait_time)
# }