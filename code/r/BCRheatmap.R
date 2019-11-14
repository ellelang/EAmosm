rm(list = ls())
library(ggplot2)
library(ggthemes)
library(maps)
library(rgdal)
library(maptools)   
library(foreign)  
library(sp)
library(tidyverse)
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")
BCRdat <- read.csv(file = "bcrdata.csv")
mosm_points <- read.csv(file = "MOSMpointsSB574.csv")
head(mosm_points )
by_sub574 <- mosm_points %>% group_by(SB574)
a <- unique(mosm_points$SB574)
b <- seq(1,574,1)
setdiff(b,a)


df_mosm574 <- by_sub574 %>% 
  summarise(
  sum_bcr = sum(SED_BCR),
  mean_bcr = mean(SED_BCR),
  n = n()) %>% 
  as.data.frame

dim(df_mosm574)

join_574 <- readOGR(dsn = "shapefiles", layer = "Join_574")
fortify(join_574)
join_574_df <- as.data.frame(join_574 )
colnames(join_574_df)

join_574_dff <- join_574_df %>% select (SB574, HydroSB,Zone, watershed, Count_, Sum_SED_BC,Avg_SED_BC,
                         Sum_Duck_B, Avg_Duck_B)


write.dbf(join_574_dff,"shapefiles/Join_574.dbf")


head(BCRdat)
str(BCRdat)
mosmPoints <- readOGR(dsn = "shapefiles", layer = "MOSMmerge_points_subinfo")
SUB574 <- readOGR(dsn = "shapefiles", layer = "SB574")
head(SUB574)
fortify(SUB574)
SUB574_df <- as.data.frame(SUB574)  
head(SUB574_df)  
  
SUB574_dff <- SUB574_df %>% 
  select(SB574, HydroSB, Zone,watershed )
write.dbf(SUB574_dff,"shapefiles/SB574.dbf")


  
## start here for lines
pts <- as.data.frame(as(mosmPoints, "SpatialPointsDataFrame"))
dim(pts)
str(pts)
pts$MOsID <- as.numeric(as.character(pts$MOsID))
pts2 = pts %>% 
  left_join(BCRdat, by = c("MOs", "MOsID" )) %>% 
  select (MOs, MOsID, Subbasin, SED_BCR, Duck_BCR,Zone, coords.x1, coords.x2)
points <- fortify(pts2)
head(points)
points
write.dbf(pts2,"shapefiles/MOSMmerge_points_subinfo.dbf")

ggplot(points, aes(x = coords.x1, y = coords.x2, group = MOs, fill = SED_BCR)) +
  geom_point(lwd=0.2)+
  theme_map()
 


# Construct a SpatialPointsDataFrame
data(mosmPoints)
xy <- mosmPoints[1]
xy
df <- mosmPoints[-1:-2]
SPDF <- SpatialPointsDataFrame(coords=xy, data=df)
