rm(list = ls())
library(ggplot2)
library(ggthemes)
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")
#setwd("D:/UWonedrive/OneDrive - UW/AAMOSM2018/ECR/20180410")

library(tidyverse)
library(scatterpie)
library(xlsx)
library(readxl)
library(directlabels)

# clean data
setwd("D:/UWonedrive/OneDrive - UW/AAMOSM2018/ECR/20180410")
data0<-read.csv(file = "mosmonetime.csv", header=TRUE)
length(data0$ID)
head(data0)
ID <- data0$ID
Generation <- data0$Generation
Outlet.Sediment <- data0$Outlet.Sediment
Cost <- data0$Cost
SedRed <- data0$SedRed
Duck <- data0$Ducks
sedbcr<- (SedRed/Cost) * 1000
duckbcr <- Duck/Cost
Type <- data0$Gene
Type
N <- length (duckbcr)
N
data1<-as.data.frame(cbind(ID,Generation,Outlet.Sediment,SedRed,Duck,Cost,sedbcr,duckbcr))
head(data1)
data1$Type <- Type
head(data1)
###########BCR lambda scenarios
ldlist<-seq(0,1,0.1)
ldlist
nlambda <- length(ldlist)
bcrnames <- vector(length = nlambda)


for (i in 1: nlambda) {
  
  l_bcr_name <- paste0("lambdaSedBcr", ldlist[i])
  bcrnames[i] <- l_bcr_name
}


X <- matrix(NA, nrow = nrow(data1), ncol = nlambda)

for (i in 1:nlambda) {
  
  X[ ,i] <- (ldlist[i] * 1000 * SedRed + (1 - ldlist[i]) * 1000 * Duck ) / Cost
  
}

X <- as.data.frame(X)
names(X) <- bcrnames
modata1 <- cbind(data1,X)
head(modata1)
length(modata1[,1])

write.csv (x = modata1, file ="mosmonetime_EA.csv", row.names = FALSE)


###############Generate BCR selected results
rm(list = ls())
ldlist<-seq(0,1,0.1)
ldlist
nlambda <- length(ldlist)

modata1<- read.csv(file= "mosmonetime_EA.csv")
N <- length (modata1[,1])
N
top <- seq (1000,N,1000) # could test with larger steps
ntop <- length(top)

bcrfunclambda <- function (dataset00, lambda, selectgen){
  
  nx <- length(dataset00[,1])
  namelist <- vector (length = ntop)
  select <- matrix (NA, nrow = nx, ncol = ntop)
  Benefit_Sed <- vector (length = ntop)
  Benefit_Duck <- vector (length = ntop)
  Cost_SedDuck <- vector (length = ntop)
  lambdaindex <- which(ldlist == lambda)
  
  for (i in 1: ntop) {
    
    TN <-  top[i]
    lambdaindex <- which (ldlist == lambda)
    #topname <- paste0("Top", TN, lambda)
    topname <- paste0(TN,"0",10*lambda)
    dataset00$topname <- rep(0, nx)
    namelist[i] <- topname
    
    
    dataset00$topname [dataset00$ID %in% top_n(dataset00, TN, dataset00[,6 +lambdaindex])$ID] <- 1
    #select <-  dataset00$topname
    select [, i] <- dataset00$topname
    Benefit_Sed [i] <- sum (select [, i] * dataset00$SedRed)
    Benefit_Duck[i] <- sum (select [, i] * dataset00$Duck)
    Cost_SedDuck[i] <- sum (select [, i] * dataset00$Cost)
    
  }
  colnames(select) <- namelist
  dataoutput <- cbind (Benefit_Sed, Benefit_Duck, Cost_SedDuck)
  colnames (dataoutput) <- c("Sediment", "Duck", "Cost")
  ifelse(selectgen == 1, return (select), return (dataoutput))
  
}

length(modata1[,1])
#bcrfunclambda (modata1,0.5,2)

writetofile <- function (lambda){
  
  selectdataframe <- bcrfunclambda (modata1,lambda,1)
  weightrankdataframe <- bcrfunclambda (modata1,lambda,2)
  
  write.csv(x =  selectdataframe, file = paste("MOSMselect_ld",lambda,".csv", sep=""), row.names = FALSE)
  write.csv(x =  weightrankdataframe, file = paste("WBCR_ld",lambda,".csv", sep=""), row.names = FALSE)
}

for (i in 1: nlambda){
  lambda <- ldlist [i]
  writetofile (lambda)
  
}

############Write to the save file

values <- c(-1, 27732.9131, 149929377.3994, 49015.6441, 897.6627, 1021.1096, 211.7990, 4626.0000, 0.0000 ,2119.0000, 517.0000 ,
            7874.0000, 537.0000, 106.0000, 1112.0000, 546.0000, 0.0000, 30969626.3368)

#trydata<-read.csv(file = "MOSMselect_ld0.4.csv")
#nrow (trydata) = 16891
#ncol(trydata) = 16
seedsnum <- 16*nlambda
N = 16891
seednames <-vector()

#Generation name
for (i in 1: nlambda) {
  
  fnbcr<-paste0 ("MOSMselect_ld", ldlist[i],".csv", sep = "" )
  
  BCRtopselect <- read.csv (file = fnbcr,check.names=F)
  
  datatrans <- t(BCRtopselect)
  
  namesseed <- colnames(BCRtopselect)
  
  seednames <- append(seednames, namesseed)
  
}

appendatafile <-  matrix (NA, nrow = seedsnum, ncol = N + length(values))

for (i in 1: nlambda) {
  
  fnbcr<-paste0 ("MOSMselect_ld", ldlist[i],".csv", sep = "" )
  
  BCRtopselect <- read.csv (file = fnbcr,check.names=F)
  
  datatrans <- t(BCRtopselect)
  
  for (j in (1: nrow(datatrans))) {
    
    appendatafile[11*(i-1)+j,] <- c(datatrans[j,],values)
    
  }
  
}

rownames(appendatafile) <- seednames

write.table ( appendatafile, file="seedfile.txt", col.names = FALSE, row.names = TRUE, append = TRUE,  sep = " ", quote = FALSE)