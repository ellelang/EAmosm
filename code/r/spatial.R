#my_packages <- c("sf","dplyr", "ggplot2", "ggthemes", "tmap", "spatstat")
rm(list = ls())
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")
library(sf)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(tmap)
library(spatstat)
library(viridisLite)
library(RColorBrewer)

mosmfile16891 <- read.csv("onetime_EA_subbasins.csv")



## Lambda scenarios function
ldscenario<-function (ld){
  
  BCRT<-matrix (NA, nrow = N, ncol = length(ld) )
  for (j in (1:length(ld))){
    BCRT[ ,j] <- bcrlambda(ld[j],sedbcr,duckbcr)
  }
  # for (i in (1:N)){
  #   for (j in (1:length(ld))){
  #     BCRT[i,j] <- bcrlambda(ld[j],sedbcr[i],duckbcr[i])
  #   }
  # }
  
  colnames(BCRT)<-c("lambdaSedbcr0", "lambdaSedbcr0.1","lambdaSedbcr0.2", "lambdaSedbcr0.3",
                    "lambdaSedbcr0.4", "lambdaSedbcr0.5","lambdaSedbcr0.6", "lambdaSedbcr0.7",
                    "lambdaSedbcr0.8", "lambdaSedbcr0.9","lambdaSedbcr1.0")
  
  return (BCRT)
  #return (BCRT) will generate the Bcrlambda for the 12 scenarios
}