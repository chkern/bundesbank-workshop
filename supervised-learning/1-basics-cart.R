### Big Data Analysis ###
### Supervised Learning I ###
### Basics and CART ###

# Setup

# install.packages("ggplot2")
# install.packages("ggmap")
# install.packages("GGally")
# install.packages("rpart")
# install.packages("partykit")
# install.packages("pdp")
# install.packages("caret")
library(ggplot2)
library(ggmap)
library(GGally)
library(rpart)
library(partykit)
library(pdp)
library(caret)

load("FrankfurtMain.Rda")

## Split data in training and test set

set.seed(7345)

train <- sample(1:nrow(fr_immo), 0.8*nrow(fr_immo))
fr_test <- fr_immo[-train,]
fr_train <- fr_immo[train,]

## Some data exploration

map <- qmap("Frankfurt, Germany", zoom = 12, maptype = "hybrid")
map + geom_point(data=fr_train, aes(x=lon, y=lat, color=rent), size=2, alpha=0.5) + scale_color_gradientn(colours=rev(heat.colors(10))) 

names(fr_train)
summary(fr_train)

cor(fr_train[,c(2:4,20)], use = "complete.obs")
ggpairs(fr_train[,c(2:4,20)], lower = list(continuous = "smooth"))

tab1 <- aggregate(fr_train$quarter, list(fr_train$quarter), length)
tab2 <- aggregate(fr_train[,c(2:4,20)], list(fr_train$quarter), mean)
cbind(tab1, tab2[,2:5])

p1 <- qplot(quarter, rent, data=fr_train, geom=c("boxplot"), fill=quarter)
p1 + theme(axis.ticks = element_blank(), axis.text.x = element_blank())

## CART

# Grow and prune tree (1-SE rule)

set.seed(6342)
f_tree <- rpart(rent ~ m2 + rooms + lon + lat + dist_to_center, data = fr_train, cp = 0.0025)
f_tree
printcp(f_tree)
plotcp(f_tree)

minx <- which.min(f_tree$cptable[,"xerror"])
minxse <- f_tree$cptable[minx,"xerror"] + f_tree$cptable[minx,"xstd"]
minse <- which.min(abs(f_tree$cptable[1:minx,"xerror"] - minxse))
mincp <- f_tree$cptable[minse,"CP"]

p_tree <- prune(f_tree, cp = mincp)
p_tree

# Variable Importance and Plots

prty_tree <- as.party(p_tree)
plot(prty_tree, gp = gpar(fontsize = 6))

varImp(p_tree)

pdp1 <- partial(p_tree, pred.var = "m2")
plotPartial(pdp1, rug = T, train = fr_train, alpha = 0.3)

pdp2 <- partial(p_tree, pred.var = c("m2", "dist_to_center"))
plotPartial(pdp2, levelplot = F, drape = T, colorkey = F, screen = list(z = 40, x = -60))

# Prediction

y_tree <- predict(p_tree, newdata = fr_test)
postResample(pred = y_tree, obs = fr_test$rent)
