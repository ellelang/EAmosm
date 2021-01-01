rm(list=ls())
wcmo<-read.csv("WCMOpoints.csv")
wcmoid<-wcmo$Site_ID
east<-log((wcmo$POINT_X)/1000)
north<-log(wcmo$POINT_Y/1000)
grassshare<-read.csv("grassshare.csv")
neweast<-wcmo$POINT_X
newnorth<-wcmo$POINT_Y
wcmograssshare<-grassshare$Percent.grass[match(wcmo$Site_ID,grassshare$WCMOid)]
wcmograssshare

# #Temporal wetland PFULL
PFULL_temp<-exp(21.794 - 2.587 * north  + 0.024 * log(sum(wcmo$shapearea)))-0.5
PFULL_temp
# 
# #Seasonal wetland PFULL
PFULL_seasonal<-exp(10.369 + 0.114 * east - 1.305 * north + 0.047 * log(sum(wcmo$shapearea)))-0.5
PFULL_seasonal

# #semipermanent wetland PFULL
PFULL_semiper<-exp(-119.705 + 18.953 * east + 13.821 * north  + -2.186 * (east * north))-0.5
PFULL_semiper
# 

sum(wcmo$shapearea)

data1<-cbind(wcmoid,wcmo$POINT_X,wcmo$POINT_Y,east,north,wcmograssshare,PFULL_semiper,PFULL_temp,PFULL_seasonal)

#datawcmopn<-read.csv(file="WCMONP.csv")
#temp_np<-0.7

wetarea_temp<-PFULL_temp*(wcmo$shapearea)
sqwetarea_temp<-sqrt(wetarea_temp)
wetarea_seasonal<-PFULL_seasonal*(wcmo$shapearea)
sqwetarea_seasonal<-sqrt(wetarea_seasonal)
wetarea_semiper<-PFULL_semiper*(wcmo$shapearea)
sqwetarea_semiper<-sqrt(wetarea_semiper)

#Mallard
#Temporary
np_temp_mallard<--0.3039*wetarea_temp+35.4917*sqwetarea_temp-0.052*sqwetarea_temp*east-0.0063*sqwetarea_temp*north+0.00000936*sqwetarea_temp*north*east
np_temp_mallard
windows(width =20, height=10)
plot(np_temp_mallard[wetarea_temp<=0.1]~wetarea_temp[wetarea_temp<=0.1])

#Seasonal
np_seasonal_mallard<-0.142*wetarea_seasonal+7.3974*sqwetarea_seasonal-0.0022*sqwetarea_seasonal*east-0.0011*sqwetarea_seasonal*north
np_seasonal_mallard
windows(width =20, height=10)
plot(np_seasonal_mallard[wetarea_seasonal<=0.1]~wetarea_seasonal[wetarea_seasonal<=0.1])

#Semipemanent
np_semiper_mallard<-0.009*wetarea_semiper+0.9034*sqwetarea_semiper-0.0019*sqwetarea_semiper*east-0.008*sqwetarea_semiper*north
np_semiper_mallard
windows(width =20, height=10)
plot(np_semiper_mallard[wetarea_semiper<=0.1]~wetarea_semiper[wetarea_semiper<=0.1])



#Gadwall
#Temporary
np_temp_gadwall<-48.5361*sqwetarea_temp-0.7706*sqwetarea_temp*east-0.0089*sqwetarea_temp*north+0.0000141*sqwetarea_temp*north*east
np_temp_gadwall
windows(width =20, height=10)
plot(np_temp_gadwall[wetarea_temp<=0.1]~wetarea_temp[wetarea_temp<=0.1])

#Seasonal
np_seasonal_gadwall<-0.1632*sqwetarea_seasonal+36.6815*sqwetarea_seasonal-0.0528*sqwetarea_seasonal*east-0.0066*sqwetarea_seasonal*north+0.00000946*sqwetarea_seasonal*north*east
np_seasonal_gadwall
windows(width =20, height=10)
plot(np_seasonal_gadwall[wetarea_seasonal<=0.1]~wetarea_seasonal[wetarea_seasonal<=0.1])

#Semipemanent
np_semiper_gadwall<-0.0171*wetarea_semiper+6.7852*sqwetarea_semiper-0.3815*sqwetarea_semiper*east-0.00883*sqwetarea_semiper*north
np_semiper_gadwall
windows(width =20, height=10)
plot(np_semiper_gadwall[wetarea_semiper<=0.1]~wetarea_semiper[wetarea_semiper<=0.1])

#Blue-winged teal
#Temporary
np_temp_bwt<-91.1876*sqwetarea_temp-0.1497*sqwetarea_temp*east-0.0168*sqwetarea_temp*north+0.0000274*sqwetarea_temp*north*east
np_temp_bwt
windows(width =20, height=10)
plot(np_temp_bwt[wetarea_temp<=0.1]~wetarea_temp[wetarea_temp<=0.1])

#Seasonal
np_seasonal_bwt<-0.3363*wetarea_seasonal+89.6838*sqwetarea_seasonal-0.0187*wetarea_seasonal*sqwetarea_seasonal- 0.1366*sqwetarea_seasonal*east-0.0164*sqwetarea_seasonal*north+0.0000251*sqwetarea_seasonal*north*east
np_seasonal_bwt
windows(width =20, height=10)
plot(np_seasonal_bwt[wetarea_seasonal<=0.1]~wetarea_seasonal[wetarea_seasonal<=0.1])

#Semipemanent
np_semiper_bwt<-0.0111*wetarea_semiper+52.5978*sqwetarea_semiper+0.000339*wetarea_semiper*sqwetarea_semiper - 0.0774*sqwetarea_semiper*east-0.0094*sqwetarea_semiper*north+0.000014*sqwetarea_semiper*north*east
np_semiper_bwt
windows(width =20, height=10)
plot(np_semiper_bwt[wetarea_semiper<=0.1]~wetarea_semiper[wetarea_semiper<=0.1])



#Northern shoveler
#Temporary
np_temp_ns<-16.1481*sqwetarea_temp-0.0238*sqwetarea_temp*east-0.0028*sqwetarea_temp*north+0.00000419*sqwetarea_temp*north*east
np_temp_ns
windows(width =20, height=10)
plot(np_temp_ns[wetarea_temp<=0.1]~wetarea_temp[wetarea_temp<=0.1])

#Seasonal
np_seasonal_ns<-0.0637*wetarea_seasonal+19.4071*sqwetarea_seasonal+0.0064*wetarea_seasonal*sqwetarea_seasonal- 0.0289*sqwetarea_seasonal*east-0.0035*sqwetarea_seasonal*north + 0.00000517*sqwetarea_seasonal*north*east
np_seasonal_ns
windows(width =20, height=10)
plot(np_seasonal_ns[wetarea_seasonal<=0.1]~wetarea_seasonal[wetarea_seasonal<=0.1])

#Semipemanent
np_semiper_ns<-0.0076*wetarea_semiper+4.6515*sqwetarea_semiper-0.0000546*wetarea_semiper*sqwetarea_semiper - 0.0022*sqwetarea_semiper*east-0.000621*sqwetarea_semiper*north
np_semiper_ns
windows(width =20, height=10)
plot(np_semiper_ns[wetarea_semiper<=0.1]~wetarea_semiper[wetarea_semiper<=0.1])


#Northern pintail
#Temporary
np_temp_np<-61.7667*sqwetarea_temp-0.095*sqwetarea_temp*east-0.0112*sqwetarea_temp*north+0.0000173*sqwetarea_temp*north*east
np_temp_np
windows(width =20, height=10)
plot(np_temp_np[wetarea_temp<=0.1]~wetarea_temp[wetarea_temp<=0.1])

#Seasonal
np_seasonal_np<-0.0269*wetarea_seasonal+24.5180*sqwetarea_seasonal- 0.0342*sqwetarea_seasonal*east-0.0044*sqwetarea_seasonal*north +0.00000601*sqwetarea_seasonal*north*east
np_seasonal_np
windows(width =20, height=10)
plot(np_seasonal_np[wetarea_seasonal<=0.1]~wetarea_seasonal[wetarea_seasonal<=0.1])

#Semipemanent
np_semiper_np<-5.0466*sqwetarea_semiper- 0.0023*sqwetarea_semiper*east-0.000694*sqwetarea_semiper*north
np_semiper_np
windows(width =20, height=10)
plot(np_semiper_np[wetarea_semiper<=0.1]~wetarea_semiper[wetarea_semiper<=0.1])


day_mallard<-35
day_gadwall<-35
day_bwt<-34
day_ns<-34
day_np<-32



##DNSi = ai + (b1 * percent grass*0.01) + (b2 * easting) + (b3 * northing) + (b4 * easting * northing)
a_mallard<-1.634
a_gadwall<-1.636
a_bwt<-1.639
a_ns<-1.640
a_np<-1.629

b1<-0.0305
b2<--0.0885
b3<--0.0125
b4<-0.1529

grass<-wcmograssshare

DSRfun<-function(a,east,north,PGRASS){
  dsr_est<-a + b1*(PGRASS/10^2) +(b2*east/10^5) + (b3*north/10^5) + (b4*east*north/10^12)
  return(dsr_est)
}

dsr_mallard<-DSRfun(a_mallard,neweast,newnorth,grass)
dsr_mallard

dsr_gadwall<-DSRfun(a_gadwall,neweast,newnorth,grass)
dsr_gadwall

dsr_bwt<-DSRfun(a_bwt,neweast,newnorth,grass)
dsr_bwt


dsr_ns<-DSRfun(a_ns,neweast,newnorth,grass)
dsr_ns

dsr_np<-DSRfun(a_np,neweast,newnorth,grass)
dsr_np


dsr_mallard
dsr_gadwall
dsr_bwt
dsr_ns
dsr_np


NSfun<-function(drs,day){
ns_est<-drs^day
return(ns_est)
}

NS_mallard<-NSfun(dsr_mallard,day_mallard)
NS_mallard

NS_gadwall<-NSfun(dsr_gadwall,day_gadwall)
NS_gadwall

NS_bwt<-NSfun(dsr_bwt,day_bwt)
NS_bwt

NS_ns<-NSfun(dsr_ns,day_ns)
NS_ns

NS_np<-NSfun(dsr_np,day_np)
NS_np

NI_mallard<-1
NI_gadwall<-0.82
NI_bwt<-0.55
NI_ns<-0.75
NI_np<-0.55


CS_mallard<-8.4
CS_gadwall<-9.9
CS_bwt<-10.2
CS_ns<-7.1
CS_np<-9.9

#Mallard
H_temp_mallard<-np_temp_mallard * NS_mallard * NI_mallard * CS_mallard
H_seasonal_mallard<-np_seasonal_mallard * NS_mallard * NI_mallard * CS_mallard
H_semiper_mallard<-np_semiper_mallard * NS_mallard * NI_mallard * CS_mallard
H_temp_mallard
H_seasonal_mallard
H_semiper_mallard  


#Gadwall
H_temp_gadwall<-np_temp_gadwall * NS_gadwall * NI_gadwall * CS_gadwall
H_seasonal_gadwall<-np_seasonal_gadwall * NS_gadwall * NI_gadwall * CS_gadwall
H_semiper_gadwall<-np_semiper_gadwall * NS_gadwall * NI_gadwall* CS_gadwall
H_temp_gadwall
H_seasonal_gadwall
H_semiper_gadwall

#Blue-winged teal
H_temp_bwt<-np_temp_bwt * NS_bwt * NI_bwt* CS_bwt
H_seasonal_bwt<-np_seasonal_bwt * NS_bwt * NI_bwt * CS_bwt
H_semiper_bwt<-np_semiper_bwt * NS_bwt* NI_bwt* CS_bwt
H_temp_bwt
H_seasonal_bwt
H_semiper_bwt


#Northern Shoveler
H_temp_ns<-np_temp_ns * NS_ns * NI_ns* CS_ns
H_seasonal_ns<-np_seasonal_ns * NS_ns * NI_ns * CS_ns
H_semiper_ns<-np_semiper_ns * NS_ns* NI_ns* CS_ns
H_temp_ns
H_seasonal_ns
H_semiper_ns

#Northern Pintail
H_temp_np<-np_temp_np * NS_np * NI_ns* CS_np
H_seasonal_np<-np_seasonal_np * NS_np * NI_np * CS_np
H_semiper_np<-np_semiper_np* NS_np* NI_np* CS_np
H_temp_np
H_seasonal_np
H_semiper_np

data1<-cbind(wcmoid,wcmo$POINT_X,wcmo$POINT_Y,east,north,wcmograssshare,PFULL_semiper,PFULL_temp,PFULL_seasonal)
data2<-cbind(wcmoid,wcmograssshare,H_temp_mallard,H_temp_gadwall,H_temp_bwt,H_temp_ns,H_temp_np,H_seasonal_mallard,H_seasonal_gadwall,H_seasonal_bwt,H_seasonal_ns,H_seasonal_np,H_semiper_mallard,H_semiper_gadwall ,H_semiper_bwt ,H_semiper_ns ,H_semiper_np)
write.csv(x=data2,file = "duck_est.csv",row.names = FALSE)

##########
library(ggplot2)
library(dplyr)
library(tidyr)
wcmo<-read.csv("WCMOpoints.csv")
duck<-read.csv("duck_est.csv")
which( colnames(wcmo)=="Site_ID" )
which( colnames(wcmo)=="shapearea" )

wcmoarea<-wcmo[,c(17,51)]
colnames(wcmoarea)[colnames(wcmoarea) == 'Site_ID'] <- "wcmoid"
joined<-left_join(duck,wcmoarea)
write.csv(x=joined,file = "duck_est2.csv",row.names = FALSE)

#############





# rm(list=ls())
# newcor<-read.csv("WCMOpoints.csv")
# wcmo<-read.csv("WCMOPOINT.csv")
# wcmoid<-wcmo$Site_ID
# east<-log((wcmo$POINT_X)/1000)
# north<-log(wcmo$POINT_Y/1000)
# grassshare<-read.csv("grassshare.csv")
# neweast<-newcor$POINT_X[match(grassshare$WCMOid,newcor$Site_ID)]
# newnorth<-newcor$POINT_Y[match(grassshare$WCMOid,newcor$Site_ID)]
# 
# wcmograssshare<-grassshare$Percent.grass[match(grassshare$WCMOid,wcmo$Site_ID)]
# wcmograssshare
# grass<-grassshare$Percent.grass
# mean(grass)
# min(grass)
# max(grass)
# ##DNSi = ai + (b1 * percent grass*0.01) + (b2 * easting) + (b3 * northing) + (b4 * easting * northing)
# a_mallard<-1.634
# a_gadwall<-1.636
# a_bwt<-1.639
# a_ns<-1.640
# a_np<-1.629
# 
# b1<-0.0305
# b2<--0.0885
# b3<--0.0125
# b4<-0.1529
# 
# sd.b1<-0.0062
# sd.b2<-0.0321
# sd.b3<-0.0030
# sd.b4<-0.0604
# 
# 
# dsreast<-wcmo$POINT_X
# hist(dsreast/100000)
# 
# min(dsreast)
# dsrnorth<-wcmo$POINT_Y
# hist(dsrnorth/100000)
# max(dsrnorth)


# 
# dsr_mallard<-a_mallard+b1*(grass/100)+b2*neweast/100000+b3*newnorth/100000+b4*(newnorth*neweast/1000000000000)
# dsr_gadwall<-a_gadwall+b1*(grass/100)+b2*neweast/100000+b3*newnorth/100000+b4*(newnorth*neweast)/1000000000000
# dsr_bwt<-a_bwt+b1*(grass/100)+b2*neweast/100000+b3*newnorth/100000+b4*(newnorth*neweast)/1000000000000
# dsr_ns<-a_ns+b1*(grass/100)+b2*neweast/100000+b3*newnorth/100000+b4*(newnorth*neweast)/1000000000000
# dsr_np<-a_np+b1*(grass/100)+b2*neweast/100000+b3*newnorth/100000+b4*(newnorth*neweast)/1000000000000
# 
# dsr_mallard
# dsr_gadwall
# dsr_blue
# dsr_nsho
# dsr_npin
# 
# a_mallard+b1*0.37/100+b2*4.75+b3*52+b4*(52*4.75/100)
# 
# a_mallard+b1*0.37/100+b2*min(dsreast/100000)+b3*max(dsrnorth/100000)+b4*(max(dsrnorth)*min(dsreast)/1000000000000)
# 
# ns_temp_mallard<-0.94^(day_mallard)
# ns_temp_mallard


#NS_temp_mallard<-dsr_mallard^(day_mallard)
#NS_temp_mallard

# #semipermanent wetland PFULL
# PFULL_semiper<-exp(-119.705 + 18.953 * east + 13.821 * north  + -2.186 * (east * north))
# 
# #Temporal wetland PFULL
# PFULL_temp<-exp(21.794 - 2.587 * north  + 0.024 * log(sum(wcmo$shapearea)))
# 
# #Seasonal wetland PFULL
# PFULL_seasonal<-exp(10.369 + 0.114 * east - 1.305 * north + 0.047 * log(sum(wcmo$shapearea)))
#                     
# data1<-cbind(wcmoid,wcmo$POINT_X,wcmo$POINT_Y,east,north,PFULL_semiper,PFULL_temp,PFULL_seasonal)
# colnames(data1) <- c("WCMOid","Xcoordinates","Ycoordinates","Easting","Northing","SemipermanentNP", "TemporalNP","SeasonalNP")
# 
# 
# write.csv(x=data1,file="WCMONP.csv",row.names = F)


