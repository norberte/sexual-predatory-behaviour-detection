library(LiblineaR)
library(readr)
library('e1071')
library(MASS)
library(glmnet)
library(caret)
library(randomForest)
library(gbm)

#min400 <- read.csv('C:/Users/Norbert/Desktop/TWE output/minDoc_dim400.csv', header = TRUE)

set.seed(2017)
samp <- sample(1:155113, 100000)

#import data
data <- read.csv('C:/Users/Norbert/Desktop/TWE output/minDoc_dim500.csv', header = TRUE)[samp,]
labels <- read.csv('C:/Users/Norbert/Desktop/TWE output/new_labels.csv', header = TRUE)[samp,]

train <-sample(1:100000, 60000)

xTrain <- data[train,]
xTest <- data[-train,]
yTrain <- labels[train]
yTest <- labels[-train]


# SVM ----------------
svm_tune2 <- tune(svm, factor(yTrain)~ .,data = xTrain , kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune2)


# Lasso --------------
grid <- 10^seq(2, -7, length=200)
grid1 <- 2^seq(5, -100, length=600)
grid2 <- 10^seq (2, -60, length =300)

b <- as.matrix(xTrain)
y <- factor(yTrain)

lasso <- cv.glmnet(b, y, alpha = 1,lambda = grid, family = "binomial")
plot(lasso$glmnet.fit, label=TRUE)
plot(lasso)
lam <- lasso$lambda.min
lamsm <- lasso$lambda.1se
lasso1_linkage2 <- glmnet(b, y, alpha=1, lambda=lamsm, family = "binomial")
lasso2_linkage2 <- glmnet(b, y, alpha=1, lambda=lam, family = "binomial")


# LibLinear SVM L2-loss
libLinModel <- LiblineaR(data = as.matrix(xTrain), target = yTrain, type = 1, verbose = TRUE)
saveRDS(libLinModel, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_LiblineaR.rds")

# SVM
svm_linear2 <- svm(factor(yTrain)~ ., data = xTrain , kernel='linear', cost=0.1, gamma = 0.5, scale=FALSE)
summary(svm_linear2)
saveRDS(svm_linear2, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_SVM.rds")

# LDA ----------------------------------------------------------
ldamod_linkage2 <- lda(factor(yTrain)~., data = xTrain)
saveRDS(ldamod_linkage2, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_LDA.rds")

# QDA ----------------------------------------------------------
qdamod_linkage2 <- qda(factor(yTrain)~., data = xTrain)
saveRDS(qdamod_linkage2, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_QDA.rds")

# Gradient Boosting ---------------
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit_linkage2 <- train(factor(yTrain)~., data = xTrain, method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)
saveRDS(grBoostFit_linkage2, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_grBoosting.rds")

# Random forrest ----------------------------------------------------------
train2RF <- randomForest(factor(yTrain)~., data = xTrain, importance=TRUE)
train2RF
saveRDS(train2RF, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_randomForest.rds")

# bagging ----------------------------------------------------------
linkage2Bag <- randomForest(factor(yTrain)~., data = xTrain, mtry=500, importance=TRUE)
linkage2Bag
saveRDS(linkage2Bag, "C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_bagging.rds")






# Predciting SVM
svm_linear2 <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_SVM.rds")
pred_svm_1 <- predict(svm_linear2, xTest)
confusionMatrix(data = pred_svm_1 , reference = yTest)

# LibLineaR predictions
libLinModel <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_LiblineaR.rds")
predictions <- predict(libLinModel, as.matrix(xTest))
confusionMatrix(data = predictions$predictions, reference = yTest)

# Predicting LDA
ldamod_linkage2 <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_LDA.rds")
pred_lda_1 <- predict(ldamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_lda_1$class, reference = yTest)

# Predicting QDA
qdamod_linkage2 <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_QDA.rds")
pred_qda_1 <- predict(qdamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_qda_1$class, reference = yTest)

# Predicting Random Forrest
train2RF <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_randomForest.rds")
pred_RF<- predict(train2RF, xTest)
confusionMatrix(data = pred_RF, reference = yTest)

# Gradient Boosting predictions
grBoostFit_linkage2 <- readRDS("C:/Users/Norbert/Desktop/Honours/R codes/classifier models/min500_grBoosting.rds")
pred_boost <- round(predict(grBoostFit_linkage2, xTest, type= "prob"))[,2]
confusionMatrix(data = pred_boost, reference = yTest)



# Bagging predictions
pred_bag <- predict(linkage2Bag, xTest)
confusionMatrix(data = pred_bag, reference = yTest)

# Predicting Lasso
newX <- as.matrix(xTest)

pred_Lasso_1 <- predict(lasso1_linkage2, newx=newX , s = lasso1_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_1 , reference = yTest)

pred_Lasso_2 <- predict(lasso2_linkage2, newx=newX , s = lasso2_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_2 , reference = yTest)
