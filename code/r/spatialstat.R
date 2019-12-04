rm(list = ls())
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")
library(sf)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggthemes)
library(tmap)
library(spdep)
library(spatstat)
library(viridisLite)
library(RColorBrewer)
library(gstat)
library(fastDummies)
sub574_sp <- st_read("shapefiles/dist_sub574.shp")
sub574_sp$SB574
names(sub574_sp)


##############distance to outlet 
dis_outlet <- read.csv("dist574_to_outlet.csv")
names(dis_outlet)

outlet_dist <- dis_outlet %>% spread(key = NEAR_FID, value = DISTANCE )
head(outlet_dist)
colnames(outlet_dist) <- c("SB574", "OLT0", "OLT1","OLT2", "OLT3","OLT4")

####################

sub574_sp_df <- sub574_sp %>% left_join(outlet_dist, by = "SB574")
sub574_sp_df

sub574_sp_df[is.na(sub574_sp_df)] <- 0
###########Moran I test
nb <- poly2nb(sub574_sp_df)
lw <- nb2listw(nb, style="B", zero.policy=TRUE)
#ld_1.lag <- lag.listw(lw, sub574_sp$disld1)

moran.test(sub574_sp_df$disld1,lw)
MCld0<- moran.mc(sub574_sp_df$disld0, lw, nsim=999)
MCld0

MCld0.5<- moran.mc(sub574_sp_df$disld05, lw, nsim=999)
MCld0.5

MCld1<- moran.mc(sub574_sp_df$disld1, lw, nsim=1000)
MCld1

################
library(olsrr)
model <- lm(disld1 ~ NEAR_DIST + Zone + HydroSB + OLT0 + OLT1 + OLT2 + OLT3 + OLT4, data = sub574_sp_df)
ols_step_best_subset(model)
## the distance to outlets are not significant

sub574_sp_df <- dummy_cols(sub574_sp_df, select_columns = c("Zone","HydroSB"),
                           remove_first_dummy = TRUE)



names(sub574_sp_df)
model_dist0 <- glm(
  disld0 ~ NEAR_DIST,  # + HydroSB + Zone , 
  data = sub574_sp_df, 
  family = "gaussian")

summary(model_dist0)

model_dist0.5 <- glm(
  disld05 ~ NEAR_DIST , # + HydroSB + Zone, 
  data = sub574_sp_df, 
  family = gaussian)

summary(model_dist0.5)

model_dist1 <- glm(
  disld1 ~ NEAR_DIST + Zone_2 + Zone_1, # Zone is more significant
  data = sub574_sp_df, 
  family =gaussian)

summary(model_dist1)
confint(model_dist1)

################### Residual autocorrelation test
sub574_sp_df$spatial_resid_glm <- residuals(model_dist1)
#plot(sub574_sp_df['spatial_resid_glm'])
moran.mc(sub574_sp_df$spatial_resid_glm, nb2listw(nb), 999)

############################### Bayesian GLM
library(R2BayesX)

#
bayes_ld1 <- bayesx(disld1 ~ NEAR_DIST,  
                    family = "gaussian", data = data.frame(sub574_sp), 
                    control = bayesx.control(seed = 170000))
summary(bayes_ld1)


############################### Bayesian GLM add spatial term 
# Compute adjacency objects 
subbasins <- st_read("shapefiles/subbasins.shp")
length(subbasins$OBJECTID)
streams <- st_read("shapefiles/LeSueur_Streams.shp")
#nb30 <- poly2nb(streams) # doesn't work
length(streams$OBJECTID)


nb30 <- poly2nb(subbasins) 
#sub_gra <- nb2gra(nb)
sub_30_gra <- nb2gra(nb30)

names(subbasins)
# Fit Bayesian spatial model ==> add a spatial term: 30 hydrology subbasin 
bayld1_spatial <- bayesx(
  disld1 ~ NEAR_DIST + sx(HydroSB, bs = "spatial", map = sub_30_gra),
  family = "gaussian", data = data.frame(sub574_sp), 
  control = bayesx.control(seed = 66100)
)

# Summarize the model
summary(bayld1_spatial)


# Map the residuals
sub574_sp$spatial_resid <- residuals(bayld1_spatial)

# Test residuals for spatial correlation: 0.999==>  can't reject the H0 that residual is not spatial autocorrelation
moran.mc(sub574_sp$spatial_resid, nb2listw(nb), 999)

# Map the fitted spatial term only
sub574_sp$spatial <- fitted(bayld1_spatial)
plot(sub574_sp['spatial'])

# plot of residuals
plot(sub574_sp['spatial_resid'])

# original map of robustness

tm_shape(sub574_sp) + tm_polygons(lwd = 0.25, "disld05", style = "jenks", n = 12) +
  tm_shape(subbasins)+tm_borders(lwd=0.5, col= "black")+
  tm_shape(streams)+tm_lines(lwd = 1, col= "cyan4") +
  tm_legend(show=FALSE)+ tm_layout(frame = FALSE, title = expression(paste(lambda, "=1")),title.size=0.6) 


#sub574_sp$spatial <- fitted(bayld1_spatial, term = "sx(HydroSB):mrf")[, "Mean"]


#############################
# variogram of the data
plot(variogram(disld0~ 1, sub574_sp))
plot(variogram(disld05~1, sub574_sp))
plot(variogram(disld1~ 1, sub574_sp))

