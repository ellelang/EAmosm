
rm(list = ls())
setwd("C:/Users/langzx/Desktop/github/EAmosm/data")
library(tidyverse)
library(ggplot2)
library(RColorBrewer)
maple29 <- read.csv (file = "maple29.csv")
cost_maple29 <- maple29$Cost
cost_sign <- sample (cost_maple29, 100)
cost_sign

x.gam<-rgamma(200,rate=0.5,shape=3.5)
med.gam<-mean(x.gam) ## sample mean
var.gam<-var(x.gam) ## sample variance
l.est<-med.gam/var.gam ## lambda estimate (corresponds to rate)
a.est<-((med.gam)^2)/var.gam ## alfa estimate
l.est
curve(dgamma(x, scale=1.5, shape=2),from=0, to=10, main="Gamma
distribution")

meds <- c(0.71, 0.731, 0.581, 0.684, 0.626, 0.611, 0.66, 0.56)
vars <- c(0.11, 0.114, 0.16, 0.126, 0.068, 0.07, 0.07, 0.137)

l_s <- meds/vars
l_s
a_s <- ((meds)^2)/vars 
a_s
xx <- rgamma(100,rate = l_s[1],shape = a_s[1])


curve(dgamma(x, scale=l_s[1], shape=a_s[1]),from=0, to=100, main="Gamma
distribution")

x_id <- seq (1, 100, 1)
y <- dgamma(x, scale=l_s[1], shape=a_s[1])*1000
plot (x, y)
dgamma(1, scale=l_s[1], shape=a_s[1])*1000
sed_sign <- y

data_exp <- data.frame (cbind (x_id, sed_sign, cost_sign))
data_exp


k = 0.8 # for linear cascade reservior
n = 1 # number of reserviors
# exact
time_range <- seq (0, 10, 0.1)
Q1 <- dgamma(time_range, shape = n, scale = k)
Q2 <- dgamma(time_range, shape = n+1, scale = k)
Q3 <- dgamma(time_range, shape = n+2, scale = k)
Q4 <- dgamma(time_range, shape = n+3, scale = k)
r <- data.frame (time_range, Q1, Q2, Q3, Q4)
r.gather <- r %>% gather (key = "alpha", value = "outflow", -time_range)
head(r.gather)


gammaplot <- ggplot(data = r.gather, mapping = aes(x = time_range, y = outflow, color=alpha))+
  geom_line(size = 2)+
  scale_color_discrete(name = expression(alpha), labels = c("1", "2", "3", "4"))+
  xlab('x') +
  ylab('f(x)') +
  ggtitle("Gamma distribution with beta = 0.8 ")
ggsave(plot = gammaplot, width = 10, height = 5, dpi = 300, filename = "gamma.png")


x <- seq(0, 50, 0.1)
Q_atp <- 1720 * x ^ 2.2 * exp(2.2*(1 - x))
Q_atp

r1 <- data.frame(x, Q_atp)

ggplot() + 
  geom_line(data = r1, aes(x = x, y = Q_atp), color = "red") 


times <- c(0, 6,12,24,60,120)
times
K = 2.4
n = 6
t_p <- 12
t_g <- n * K
Q_P = 1720
Q_outflow <- Q_P *((times/t_p)^(t_p/(t_g - times)) * exp((t_p - times)/(t_g - t_p)))
Q_outflow

o <- data.frame(times, Q_outflow)

#diff <- ifelse (Q1-Q2 >= 0, Q1-Q2, 0)
diff <- Q1-Q2

plot (time_range, diff, type = "l")
