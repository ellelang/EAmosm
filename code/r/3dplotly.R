rm(list=ls())
library(plotly)
library(tidyverse)
#library(readxl)
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")


Sys.setenv("plotly_username"="ellelang")
Sys.setenv("plotly_api_key"="A1qQ7mXcqBA3EbeVXsUm")
# linearized greedy frontier

greedy_front<- read.csv (file = "ldfront_seeds.csv")

p <- plot_ly(greedy_front, x = ~SRed, y = ~Duck, z = ~Cost,
             marker = list(color = 'red', colorscale = c('#FFE1A1', '#683531'), showscale = TRUE)) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Sediment reduction'),
                      yaxis = list(title = 'Duck'),
                      zaxis = list(title = 'Cost')),
         annotations = list(
           x = 1.0,
           y = 1.0,
           text = 'text',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
p
htmlwidgets::saveWidget(as_widget(p), file = "greedy_front.html")
