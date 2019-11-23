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
names(mosmfile16891)

ld <- seq(0,1,0.1)
ld
ldname <- paste("lambdased",ld, sep = "" )
ldname
mosmfile16891[ldname[1]] = mosmfile16891$SED_BCR_x * ld[1] + mosmfile16891$Duck_BCR_x * (1-ld[1])

for (i in 1:length(ld)){
  mosmfile16891[ldname[i]] = mosmfile16891$SED_BCR_x * ld[i] + mosmfile16891$Duck_BCR_x * (1-ld[i])
}
names(mosmfile16891)
head(mosmfile16891$lambdased0.1)

dist_df <- mosmfile16891 %>% 
  arrange(desc(lambdased1)) %>% 
  mutate(dist_ldsed1 = abs(lambdased1 - lag(lambdased1, default = lambdased1[1]))*1000)%>% 
  arrange(desc(lambdased0)) %>% 
  mutate(dist_ldsed0 = abs(lambdased0 - lag(lambdased0, default = lambdased0[1]))*1000)%>%
  arrange(desc(lambdased0.5)) %>% 
  mutate(dist_ldsed0.5 = abs(lambdased0.5 - lag(lambdased0.5, default = lambdased0.5[1]))*1000)

names(dist_df)
dist_df$dist_ldsed0
dist_df$dist_ldsed1- dist_df$dist_ldsed0.5


mosmpoints_sp <- st_read("shapefiles/MOSMpointsSB574.shp")
head(mosmpoints_sp)
dist_df_prep <- dist_df %>% 
  select ("MOs","MOsID","SED_B","SED_BCR_x","Duck_BCR_x","dist_ldsed1","dist_ldsed0","dist_ldsed0.5") %>% 
  left_join(mosmpoints_sp, by.x = c("MOs","MOsID"), by.y = c("MOs","MOsID"))

dist_df_prep$SB574

sub574_dist <- as.data.frame(dist_df_prep %>% 
    group_by(SB574)%>% 
    summarise(
    sedred_mean = mean(SED_B,na.rm=TRUE),
    sedbcr_mean = mean(SED_BCR_x,na.rm=TRUE),
    distldsed0_mean = mean(dist_ldsed0,na.rm=TRUE),
    distldsed0.5_mean = mean(dist_ldsed0.5,na.rm=TRUE),
    distldsed1_mean = mean(dist_ldsed1,na.rm=TRUE), 
  ))



sub574_sp <- st_read("shapefiles/SB574.shp")

sub574_sp <- sub574_sp %>% left_join(sub574_dist, by.x = SB574, by.y = SB574)

names(sub574_sp)
#st_write(sub574_sp, "shapefiles/dist_sub574.shp", driver="ESRI Shapefile") 

tm_shape(sub574_sp) + 
  tm_polygons("distldsed0_mean", palette = "Greens", 
              style = "quantile", n = 12, 
              title = "Focus on Sediment Reduction")


sub574_join <- st_read("shapefiles/SB574_bcrdist.shp")

tm_shape(sub574_join) + 
  tm_polygons("sedbcr_dis", palette = "Greens", 
              style = "quantile", n = 12, 
              title = "Focus on Sediment Reduction")

st_crs(sub574_sp)
#############test
library(spdep)
sub574_sp[is.na(sub574_sp)] <- 0

nb <- poly2nb(sub574_sp)
lw <- nb2listw(nb, style="B", zero.policy=TRUE)
ld_1.lag <- lag.listw(lw, sub574_sp$distldsed1_mean)

# Create a regression model
M <- lm(ld_1.lag ~ sub574_sp$distldsed1_mean)

# Plot the data
plot( ld_1.lag ~ sub574_sp$distldsed0.5_mean, pch=20, asp=1, las=1)

moran.test(sub574_sp$distldsed0.5_mean,lw)
MC<- moran.mc(sub574_sp$distldsed1_mean, lw, nsim=999)
MC

#####
sub574_cenpoints <- st_centroid(sub574_sp)
streams <- st_read("shapefiles/LeSueur_Streams.shp")
subbasins <- st_read("shapefiles/subbasins.shp")

sb30 = st_geometry(subbasins)
plot(sb30, border = 'grey')
plot(streams, add = TRUE)
plot(st_centroid(pol), add = T)
#dist <- st_distance(sub574_cenpoints, streams, by_element = TRUE)



model_dist <- glm(
  distldsed0_mean ~ NEAR_DIST, 
  data = sub574_sp, 
  family =gaussian)

summary(model_dist)

ggplot(sub574_sp) + 
  geom_sf(aes(colour = NEAR_DIST))+
  scale_colour_gradient2()

#spplot(sub574_sp, zcol = "NEAR_DIST")


tm1 <- tm_shape(sub574_sp) + tm_polygons(lwd = 0.005, "distldsed1_mean", style = "jenks", n = 12) +
  tm_shape(subbasins)+tm_borders(lwd=0.5, col= "black")+
  tm_shape(streams)+tm_lines(lwd = 1, col= "cyan4") +
  tm_legend(show=FALSE)+ tm_layout(frame = FALSE, title = expression(paste(lambda, "=1")),title.size=0.6) 

tm05 <- tm_shape(sub574_sp) + tm_polygons(lwd = 0.005, "distldsed0.5_mean", style = "jenks", n = 12) +
  tm_shape(subbasins)+tm_borders(lwd=0.5, col= "black")+
  tm_shape(streams)+tm_lines(lwd = 1, col= "cyan4") +
  tm_legend(show=FALSE)+ tm_layout(frame = FALSE, title = expression(paste(lambda, "=0.5")),title.size=0.6) 

tm0 <- tm_shape(sub574_sp) + tm_polygons(lwd = 0.005, "distldsed0_mean", style = "jenks", n = 12) +
  tm_shape(subbasins)+tm_borders(lwd=0.5, col= "black")+
  tm_shape(streams)+tm_lines(lwd = 1, col= "cyan4") +
  tm_legend(show=TRUE)+ tm_layout(frame = FALSE, legend.text.size = 0.6, title = expression(paste(lambda, "=0")),
                                  title.size=0.6) 


m2 <- tmap_arrange(tm1, tm05, tm0,nrow = 1, ncol = 3)

#tmap_save(m2, "Robust_map.png", width=1920, height=1080, asp=0)
tmap_save(m2, "Robust_map.png", width=2920, height=1080, dpi = 600)

library(grid)
grid.newpage()
pushViewport(viewport(layout=grid.layout(1,3)))
print(tm1, vp=viewport(layout.pos.col = 1))
print(tm05, vp=viewport(layout.pos.col = 2))
print(tm0, vp=viewport(layout.pos.col = 3))



plot(st_buffer(u, 0.2))
plot(u, border = 'red', add = TRUE)
plot(st_buffer(u, 0.2), border = 'grey')
plot(u, border = 'red', add = TRUE)
plot(st_buffer(u, -0.2), add = TRUE)