library(LiblineaR)
library(readr)

#import data
labels <- read.csv('C:/Users/Norbert/Desktop/TWE output/new_labels.csv', header = TRUE)

min400 <- read.csv('C:/Users/Norbert/Desktop/TWE output/minDoc_dim400.csv', header = TRUE)
max400 <- read.csv('C:/Users/Norbert/Desktop/TWE output/maxDoc_dim400.csv', header = TRUE)

train <-sample(1:dim(max400)[1],100000)

xTrain <- max400[train,]
xTest <- max400[-train,]
yTrain <- labels[train,]
yTest <- labels[-train,]


# SVM ----------------
library('e1071')
svm_tune2 <- tune(svm, factor(yTrain)~ .,data = xTrain , kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune2)

svm_linear2 <- svm(factor(yTrain)~ ., data = xTrain , kernel='linear', cost=0.1, gamma = 0.5, scale=FALSE)
summary(svm_linear2)

# Predciting SVM
pred_svm_1 <- predict(svm_linear2, xTest)
confusionMatrix(data = pred_svm_1 , reference = yTest)


# Lasso --------------
library(glmnet)
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

# Predicting Lasso
newX <- as.matrix(xTest)

pred_Lasso_1 <- predict(lasso1_linkage2, newx=newX , s = lasso1_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_1 , reference = yTest)

pred_Lasso_2 <- predict(lasso2_linkage2, newx=newX , s = lasso2_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_2 , reference = yTest)


# LibLinear SVM L2-loss
result <- LiblineaR(data = as.matrix(xTrain), target = yTrain, type = 1, verbose = TRUE)
predictions <- predict(result, as.matrix(xTest))
confusionMatrix(data = predictions$class, reference = yTest)


# LDA ----------------------------------------------------------
library(MASS)
ldamod_linkage2 <- lda(factor(yTrain)~., data = xTrain)

# Predicting LDA
pred_lda_1 <- predict(ldamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_lda_1$class, reference = yTest)

# QDA ----------------------------------------------------------
qdamod_linkage2 <- qda(factor(yTrain)~., data = xTrain)

# Predicting QDA
pred_qda_1 <- predict(qdamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_qda_1$class, reference = yTest)

# Random forrest ----------------------------------------------------------
library(randomForest)
train2RF <- randomForest(factor(yTrain)~., data = xTrain, importance=TRUE)
train2RF

# Predicting Random Forrest
pred_RF<- predict(train2RF, xTest)
confusionMatrix(data = pred_RF, reference = yTest)

# bagging ----------------------------------------------------------
linkage2Bag <- randomForest(factor(yTrain)~., data = xTrain, mtry=400, importance=TRUE)
linkage2Bag

# Bagging predictions
pred_bag <- predict(linkage2Bag, xTest)
confusionMatrix(data = pred_bag, reference = yTest)

# Gradient Boosting ---------------
library(caret)
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit_linkage2 <- train(factor(Classification)~., data = linkage2, method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)

# Gradient Boosting predictions
pred_boost <- round(predict(grBoostFit_linkage2, xTest, type= "prob"))[,2]
confusionMatrix(data = pred_boost, reference = yTest)
