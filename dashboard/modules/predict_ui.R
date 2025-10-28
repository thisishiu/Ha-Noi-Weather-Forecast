library(shiny)
library(shinydashboard)
library(fontawesome)

predictTab <- function(tabName){
    return(
        tabItem(tabName = tabName,
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
                    inputId = "predict_selected_model", 
                    "Select model:", 
                    list("TFT" = "TFT")
                    )
                ),
              
                 column(
                    width = 3,
                    selectInput(
                        "predict_selected_region", 
                        "Select region:", 
                        district_list
                    )
                )
            ),
            
            # --- 3 valueBox + biểu đồ ---
            tags$div(
                h3("Model Performance Metrics",
                    style = "margin-top: 25px; text-align: center; font-weight: 700;
                            font-size: 28px; color: #2b2b2b; letter-spacing: 1px;"
                ),

                fluidRow(
                    style = "display: flex; align-items: stretch; gap: 20px; margin-top: 25px;",
          
                    # --- 3 box trái ---
                    column(
                        width = 3,
                        style = "display: flex; flex-direction: column; gap: 20px;",
                        
                        # Box R²
                        tags$div(
                            class = "predict-metric-box",
                            style = paste0(
                                "background: linear-gradient(145deg, #d6eaff, #74b9ff);",
                                "border-radius: 20px; padding: 20px; text-align:center;",
                                "box-shadow: 0 4px 10px rgba(0,0,0,0.15); transition: all 0.3s ease;"
                            ),
                            tags$i(class = "fas fa-chart-line", style = "font-size:45px; color:#0c2461;"),
                            tags$h4(
                                HTML("R<sup>2</sup> = 0.873"), 
                                style = "color:#0c2461; font-weight:700; margin-top:10px;
                            "),
                            tags$p(
                                "Goodness of Fit", style = "color:gray; font-size:14px; margin-top:5px;"
                            )
                        ),
                        
                        # Box MSE
                        tags$div(
                            class = "predict-metric-box",
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
                            class = "predict-metric-box",
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
                        tags$h4(
                                style = "font-weight:600; margin-bottom:15px; color:#2b2b2b; text-align:center;"),
                        plotOutput("predict_plot", height = "350px")
                        )
                    )
                ),
        
                tags$style(HTML("
                .predict-metric-box:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                }
                "))
            ),

            
            # --- Dự báo 6 giờ tới ---
            # predict_ui.R
            tags$div(
                h3("12-Hour Weather Forecast",
                    style = "margin-top: 25px; text-align: center; font-weight: 700;
                            font-size: 28px; color: #2b2b2b; letter-spacing: 1px;"),

                # Khu vực để server cập nhật
                uiOutput("predict_forecast_ui"),

                tags$p("⬅️ Scroll để xem dự báo 12 giờ tiếp theo ➡️",
                    style = "text-align: center; color: #7f8c8d; font-size: 14px; margin-top: 10px;")
            ),

            textOutput("testpredict")


        )
    )
}
