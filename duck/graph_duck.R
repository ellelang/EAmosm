rm(list=ls())
library(RColorBrewer)
#library(plotly)
mypalette1<-brewer.pal(9,"Reds")
mypalette1[1]
mypalette3<-brewer.pal(7,"Blues")
mypalette3
colfunc<-colorRampPalette(mypalette3)

data<-read.csv(file="duck_est2.csv")
area<-data$shapearea
Mallard_semipermanent<-data$H_semiper_mallard
Gadwall_semipermanent<-data$H_semiper_gadwall
BlueTeal_semipermanent<-data$H_semiper_bwt
Northernshoveler_semipermanent<-data$H_semiper_ns
Northernpintail_semipermanent<-data$H_semiper_np

x<-cbind(Mallard_semipermanent,Gadwall_semipermanent,BlueTeal_semipermanent,Northernshoveler_semipermanent,Northernpintail_semipermanent)
dim(x)
sumduck <- apply(x, 1, sum)
sumduck



windows(width=100,height=200)
mat<-matrix(c(1,0,2,
              1,0,3,
              1,0,4,
              1,0,5,
              1,0,6),nrow=5,ncol=3,byrow=T)
par(mar=c(1,1,1,1),oma=c(5,5.5,3,1))
layout(mat=mat,widths=c(10,0.1,4))
layout.show(n=6)





plot(sumduck~area,col=(colfunc(2000)),lwd=6, lty =1, xaxs = "i", yaxs = "i", ann=F)
text(x=0.3,y=150,labels="All Species",font=2,cex=3)
#mtext(side=3,outer = T, LINE = 1, "All species")
mtext(side =2, outer = T, line = 3, "Duck hatchlings",cex=1.5)
mtext(side = 1, outer = T, line = 3, "Area (KM^2)",cex=1.5)

plot(Mallard_semipermanent~area,col= (colfunc(2000)),lty =1,lwd=6, xaxs = "i", yaxs = "i",xaxt = "n" , ann=F,cex=0.5)
text(x=0.3,y=2.3,labels="Mallard",font=2,cex=1)

#mtext(side=3,outer = F, LINE = -0.1, "Each species")
plot(Gadwall_semipermanent~area,col=(colfunc(2000)),lty =1,lwd=6, xaxs = "i", yaxs = "i",xaxt = "n" , ann=F)
text(x=0.3,y=12,labels="Gadwall",font=2,cex=1)
plot(BlueTeal_semipermanent~area,col=(colfunc(2000)),lty =1,lwd=6, xaxs = "i", yaxs = "i",xaxt = "n" , ann=F)
text(x=0.3,y=120,labels="Gadwall",font=2,cex=1)

plot(Northernshoveler_semipermanent~area,col=(colfunc(2000)),lty =1,lwd=6, xaxs = "i", yaxs = "i",xaxt = "n" , ann=F)
text(x=0.4,y=10,labels="Northen shoveler",font=2,cex=1)

plot(Northernpintail_semipermanent~area,col=(colfunc(2000)),lty =1,lwd=6, xaxs = "i", yaxs = "i",xlab = "Area" , ann=F)
text(x=0.4,y=8,labels="Northen pintail",font=2,cex=1)
