library(shiny)
library(shinydashboard)
library(fontawesome) # để dùng icon nắng/mưa đẹp

predictTab <- function(tabName){
  return(
    tabItem(tabName = tabName,
      tags$head(
        tags$link(rel = "stylesheet", type = "text/css", href = "dashboard/www/home.css")
      ),
      
      # --- Tiêu đề ---
      tags$div(
        h1("Weather Prediction Section"),
        style = "text-align: center; margin-bottom: 25px; font-weight: bold; color: #333;"
      ),
      
      # --- Hàng chọn model & khu vực ---
      fluidRow(
        style = "display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;",
        
        column(
          width = 3,
          selectInput(
            "selectModel", 
            "Select model:", 
            list("A" = "A", "Choice 1B" = "1B", "Choice 1C" = "1C")
          )
        ),
        
        column(
          width = 3,
          selectInput(
            "selectRegion", 
            "Select region:", 
            list("Ba Đình" = "Ba Đình", "Hoàn Kiếm" = "Hoàn Kiếm", "Đống Đa" = "Đống Đa", "Cầu Giấy" = "Cầu Giấy")
          )
        )
      ),
      
      # --- 3 valueBox + biểu đồ ---
      tags$div(
  h3("📊 Model Performance Metrics",
     style = "margin-top: 25px; text-align: center; font-weight: 700;
              font-size: 28px; color: #2b2b2b; letter-spacing: 1px;"),

  fluidRow(
    style = "display: flex; align-items: stretch; gap: 20px; margin-top: 25px;",
    
    # --- 3 box trái ---
    column(
      width = 3,
      style = "display: flex; flex-direction: column; gap: 20px;",
      
      # Box R²
      tags$div(
        class = "metric-box",
        style = paste0(
          "background: linear-gradient(145deg, #d6eaff, #74b9ff);",
          "border-radius: 20px; padding: 20px; text-align:center;",
          "box-shadow: 0 4px 10px rgba(0,0,0,0.15); transition: all 0.3s ease;"
        ),
        tags$i(class = "fas fa-chart-line", style = "font-size:45px; color:#0c2461;"),
        tags$h4(HTML("R<sup>2</sup> = 0.873"), style = "color:#0c2461; font-weight:700; margin-top:10px;"),
        tags$p("Goodness of Fit", style = "color:gray; font-size:14px; margin-top:5px;")
      ),
      
      # Box MSE
      tags$div(
        class = "metric-box",
        style = paste0(
          "background: linear-gradient(145deg, #fff8e1, #ffeaa7);",
          "border-radius: 20px; padding: 20px; text-align:center;",
          "box-shadow: 0 4px 10px rgba(0,0,0,0.15); transition: all 0.3s ease;"
        ),
        tags$i(class = "fas fa-bullseye", style = "font-size:45px; color:#e67e22;"),
        tags$h4("MSE = 0.012", style = "color:#e67e22; font-weight:700; margin-top:10px;"),
        tags$p("Mean Squared Error", style = "color:gray; font-size:14px; margin-top:5px;")
      ),
      
      # Box RMSE
      tags$div(
        class = "metric-box",
        style = paste0(
          "background: linear-gradient(145deg, #ffe6e6, #ff7675);",
          "border-radius: 20px; padding: 20px; text-align:center;",
          "box-shadow: 0 4px 10px rgba(0,0,0,0.15); transition: all 0.3s ease;"
        ),
        tags$i(class = "fas fa-ruler", style = "font-size:45px; color:#b33939;"),
        tags$h4("RMSE = 0.109", style = "color:#b33939; font-weight:700; margin-top:10px;"),
        tags$p("Root Mean Squared Error", style = "color:gray; font-size:14px; margin-top:5px;")
      )
    ),
    
    # --- Cột phải: Biểu đồ ---
    column(
      width = 9,
      tags$div(
        style = paste0(
          "background:white; border-radius:20px; padding:20px;",
          "box-shadow: 0 4px 15px rgba(0,0,0,0.1); height:100%;"
        ),
        tags$h4("Prediction vs Actual",
                style = "font-weight:600; margin-bottom:15px; color:#2b2b2b; text-align:center;"),
        plotOutput("predictPlot", height = "350px")
      )
    )
  ),
  
  # --- CSS chung cho hiệu ứng hover ---
  tags$style(HTML("
    .metric-box:hover {
      transform: translateY(-5px);
      box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
  "))
),

      
      # --- Dự báo 6 giờ tới ---
        tags$div(
        h3("🌦 6-Hour Weather Forecast",
            style = "margin-top: 25px; text-align: center; font-weight: 700;
                    font-size: 28px; color: #2b2b2b; letter-spacing: 1px;"),
        
        fluidRow(
            style = "display: flex; justify-content: space-evenly; gap: 20px; margin-top: 25px;",
            
            lapply(1:6, function(i) {
            predict_probs <- c(0.2, 0.6, 0.1, 0.8, 0.4, 0.7)
            rain_threshold <- 0.5
            
            prob <- predict_probs[i]
            is_rain <- prob > rain_threshold
            
            icon_class <- ifelse(is_rain, "fas fa-cloud-rain", "fas fa-sun")
            box_color <- ifelse(is_rain, "#74b9ff", "#ffeaa7")
            text_color <- ifelse(is_rain, "#0c2461", "#e67e22")
            bg_gradient <- ifelse(is_rain,
                                    "linear-gradient(145deg, #d6eaff, #74b9ff)",
                                    "linear-gradient(145deg, #fff8e1, #ffeaa7)")
            
            weather_text <- ifelse(is_rain,
                                    paste0("Rainy — ", round(prob * 100), "%"),
                                    paste0("Sunny — ", round((1 - prob) * 100), "%"))
            
            column(
                width = 2,
                tags$div(
                style = paste0(
                    "background:", bg_gradient, ";",
                    "border-radius: 20px;",
                    "box-shadow: 0 4px 10px rgba(0,0,0,0.15);",
                    "padding: 20px; text-align:center;",
                    "transition: all 0.3s ease-in-out;"
                ),
                # Hiệu ứng hover
                tags$script(HTML("
                    $(document).on('mouseenter', '.weather-box', function() {
                    $(this).css({'transform': 'translateY(-5px)', 'box-shadow': '0 8px 20px rgba(0,0,0,0.2)'});
                    }).on('mouseleave', '.weather-box', function() {
                    $(this).css({'transform': 'translateY(0)', 'box-shadow': '0 4px 10px rgba(0,0,0,0.15)'});
                    });
                ")),
                class = "weather-box",
                
                tags$i(class = icon_class, style = paste0("font-size: 45px; color:", text_color, ";")),
                tags$h4(weather_text, style = paste0("color:", text_color, "; font-weight: 600; margin-top: 10px;")),
                tags$p(paste0("+", i, "h"), style = "color: gray; font-size: 14px; margin-top: 5px;")
                )
            )
            })
        )
        )

    )
  )
}
