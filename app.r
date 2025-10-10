# app.R
# install.packages(c("shiny", "ggplot2", "DT"))

library(shiny)
library(ggplot2)
library(DT)   # để hiển thị bảng đẹp

ui <- fluidPage(
  titlePanel("Ví dụ Shiny cơ bản — mtcars"),
  sidebarLayout(
    sidebarPanel(
      selectInput("xvar", "Chọn biến trục X:",
                  choices = names(mtcars), selected = "wt"),
      selectInput("yvar", "Chọn biến trục Y:",
                  choices = names(mtcars), selected = "mpg"),
      sliderInput("pointSize", "Kích thước điểm:", min = 1, max = 5, value = 2, step = 0.5),
      checkboxInput("showSmooth", "Hiện đường hồi quy (geom_smooth)", value = TRUE),
      checkboxInput("showLabels", "Hiện nhãn điểm (tên xe)", value = FALSE),
      hr(),
      downloadButton("downloadData", "Tải CSV (dữ liệu lọc)")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Biểu đồ", plotOutput("scatterPlot", height = "500px")),
        tabPanel("Bảng dữ liệu", DTOutput("table")),
        tabPanel("Thông tin", verbatimTextOutput("summary"))
      )
    )
  )
)

server <- function(input, output, session) {

  # Reactive: dữ liệu được chọn (có thể thêm filter nếu muốn)
  filteredData <- reactive({
    df <- mtcars
    df$model <- rownames(df)   # thêm tên xe làm cột
    # ta có thể thêm filter ở đây nếu cần
    df
  })

  # Plot
  output$scatterPlot <- renderPlot({
    df <- filteredData()
    x <- df[[input$xvar]]
    y <- df[[input$yvar]]

    p <- ggplot(df, aes_string(x = input$xvar, y = input$yvar)) +
      geom_point(size = input$pointSize) +
      theme_minimal() +
      labs(x = input$xvar, y = input$yvar,
           title = paste("Scatter:", input$yvar, "vs", input$xvar))

    if (input$showSmooth) {
      p <- p + geom_smooth(method = "lm", se = TRUE)
    }
    if (input$showLabels) {
      p <- p + geom_text(aes(label = model), hjust = -0.2, vjust = 0.2, size = 3)
    }
    print(p)
  })

  # Bảng dữ liệu
  output$table <- renderDT({
    df <- filteredData()
    datatable(df, options = list(pageLength = 10, scrollX = TRUE))
  })

  # Summary text
  output$summary <- renderPrint({
    df <- filteredData()
    cat("Số quan sát:", nrow(df), "\n")
    cat("Các biến hiện có:\n")
    print(names(df))
    cat("\nMô tả nhanh cho hai biến đã chọn:\n")
    print(summary(df[, c(input$xvar, input$yvar)]))
  })

  # Download handler
  output$downloadData <- downloadHandler(
    filename = function() {
      paste0("mtcars_filtered_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(filteredData(), file, row.names = TRUE)
    }
  )
}

shinyApp(ui, server)
