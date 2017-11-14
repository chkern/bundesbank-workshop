### Big Data Analysis ###
### Supervised Learning II ###
### Bagging, Random Forests and Boosting ###

# Setup

# install.packages("foreach")
# install.packages("caret")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("xgboost")
# install.packages("pdp")
library(foreach)
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(pdp)

load("FrankfurtMain.Rda")
fr_immo$address <- NULL
fr_immo$quarter <- NULL

## Split data in training and test set

set.seed(7345)

train <- sample(1:nrow(fr_immo), 0.8*nrow(fr_immo))
fr_test <- fr_immo[-train,]
fr_train <- fr_immo[train,]

## Bagging

# Using foreach

y_tbag <- foreach(m = 1:100, .combine = cbind) %do% { 
  rows <- sample(nrow(fr_train), replace = T)
  fit <- rpart(rent ~ ., data = fr_train[rows,], cp = 0.001)
  predict(fit, newdata = fr_test)
}

postResample(y_tbag[,1], fr_test$rent)
postResample(rowMeans(y_tbag), fr_test$rent)
summary(apply(y_tbag,1,var))

y_rbag <- foreach(m = 1:100, .combine = cbind) %do% { 
  rows <- sample(nrow(fr_train), replace = T)
  fit <- lm(rent ~ ., data = fr_train[rows,])
  predict(fit, newdata = fr_test)
}

postResample(y_rbag[,1], fr_test$rent)
postResample(rowMeans(y_rbag), fr_test$rent)
summary(apply(y_rbag,1,var))

# Using caret

ctrl  <- trainControl(method = "cv",
                      number = 5)

set.seed(7324)
bag <- train(rent ~ .,
             data = fr_train,
             method = "rf",
             trControl = ctrl,
             tuneGrid = data.frame(mtry = 17),
             importance = TRUE)

bag
plot(bag$finalModel)
varImp(bag)

getTree(bag$finalModel, k = 1, labelVar = T)[1:10,]
getTree(bag$finalModel, k = 2, labelVar = T)[1:10,]

## Random Forest

set.seed(7324)
rf <- train(rent ~ .,
            data = fr_train,
            method = "rf",
            trControl = ctrl,
            importance = TRUE)

rf
plot(rf)
plot(rf$finalModel)
varImp(rf)

# Inspect Forest

getTree(rf$finalModel, k = 1, labelVar = T)[1:10,]
getTree(rf$finalModel, k = 2, labelVar = T)[1:10,]

pdp3 <- partial(rf, pred.var = "m2", ice = T, trim.outliers = T)
pdp4 <- partial(rf, pred.var = "dist_to_center", ice = T, trim.outliers = T)
p1 <- plotPartial(pdp3, rug = T, train = fr_train, alpha = 0.3)
p2 <- plotPartial(pdp4, rug = T, train = fr_train, alpha = 0.3)
grid.arrange(p1, p2, ncol = 2)

pdp5 <- partial(rf, pred.var = c("lat", "lon"))
plotPartial(pdp5, levelplot = F, drape = T, colorkey = F, screen = list(z = 130, x = -60))

## Boosting

grid <- expand.grid(max_depth = 1:3,
                    nrounds = c(500, 1000),
                    eta = c(0.05, 0.01),
                    min_child_weight = 5,
                    subsample = 0.7,
                    gamma = 0,
                    colsample_bytree = 1)

grid

set.seed(7324)
xgb <- train(rent ~ .,
             data = fr_train,
             method = "xgbTree",
             trControl = ctrl,
             tuneGrid = grid)

xgb
plot(xgb)
varImp(xgb)

## CART

grid <- expand.grid(maxdepth = 1:30)
                    
set.seed(7324)
cart <- train(rent ~ .,
              data = fr_train,
              method = "rpart2",
              trControl = ctrl,
              tuneGrid = grid)

cart
plot(cart)
varImp(cart)

## Linear regression

set.seed(7324)
reg <- train(rent ~ .,
             data = fr_train,
             method = "glm",
             trControl = ctrl)

reg
summary(reg)
varImp(reg)

## Comparison (rf, xgboost, rpart & glm)

resamps <- resamples(list(Bagging = bag,
                          RandomForest = rf,
                          Boosting = xgb,
                          CART = cart,
                          Regression = reg))

resamps
summary(resamps)
bwplot(resamps, metric = c("RMSE", "Rsquared"), scales = list(relation = "free"), xlim = list(c(0, 500), c(0, 1)))
splom(resamps, metric = "RMSE")
splom(resamps, metric = "Rsquared")

difValues <- diff(resamps)
summary(difValues)

## Prediction

y_bag <- predict(bag, newdata = fr_test)
y_rf <- predict(rf, newdata = fr_test)
y_xgb <- predict(xgb, newdata = fr_test)
y_cart <- predict(cart, newdata = fr_test)
y_reg <- predict(reg, newdata = fr_test)

postResample(pred = y_bag, obs = fr_test$rent)
postResample(pred = y_rf, obs = fr_test$rent)
postResample(pred = y_xgb, obs = fr_test$rent)
postResample(pred = y_cart, obs = fr_test$rent)
postResample(pred = y_reg, obs = fr_test$rent)