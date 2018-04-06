#library(LiblineaR)
library(readr)
#library('e1071')
#library(MASS)
library(caret)
library(gbm)
#library(randomForest)

set.seed(2017)

#sample data
sam <- sample(1:155113, 100000)
data <- read.csv("minDoc_dim200.csv", header = TRUE)[sam,]
label <- read.csv("new_labels.csv", header = TRUE)[sam,]

# split 60k/40k train/test set
train <- sample(1:100000, 60000)
xTrain <- data[train,]
xTest <- data[-train,]
print("imported data")

yTrain <- label[train]
yTest <- label[-train]
print("imported labels")

# SVM ----------------
#svm_linear2 <- svm(factor(yTrain)~ ., data = xTrain , kernel='linear', cost=0.1, gamma = 0.5, scale=FALSE)
#summary(svm_linear2)
#saveRDS(svm_linear2, "min200_svm.rds")
#print("SVM finished training...")

# LibLinear SVM L2-loss
#libLinModel <- LiblineaR(data = as.matrix(xTrain), target = yTrain, type = 1, verbose = TRUE)
#saveRDS(libLinModel, "min200_liblinear.rds")
#print("LibLineaR finished training...")

# LDA ----------------------------------------------------------
#ldamod_linkage2 <- lda(factor(yTrain)~., data = xTrain)
#saveRDS(ldamod_linkage2, "min200_lda.rds")
#print("LDA finished training...")

# QDA ----------------------------------------------------------
#qdamod_linkage2 <- qda(factor(yTrain)~., data = xTrain)
#saveRDS(qdamod_linkage2, "min200_qda.rds")
#print("QDA finished training...")

# Random forrest ----------------------------------------------------------
#train2RF <- randomForest(factor(yTrain)~., data = xTrain, importance=TRUE)
#train2RF
#saveRDS(train2RF,"min200_randomForest.rds")
#print("Random Forest finished training...")

# bagging ----------------------------------------------------------
#linkage2Bag <- randomForest(factor(yTrain)~., data = xTrain, mtry=200, importance=TRUE)
#linkage2Bag
#saveRDS(linkage2Bag, "min200_bagging.rds")
#print("Bagging finished training...")

# Gradient Boosting ---------------
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit_linkage2 <- train(factor(yTrain)~., data = xTrain, method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)
saveRDS(grBoostFit_linkage2, "min200_grBoosting.rds")
print("Gradient Boosting finished training...")


print("SVM predictions...")
# Predciting SVM
pred_svm_1 <- predict(svm_linear2, xTest)
confusionMatrix(data = pred_svm_1 , reference = yTest)

# Predicting Lasso
#newX <- as.matrix(xTest)

#pred_Lasso_1 <- predict(lasso1_linkage2, newx=newX , s = lasso1_linkage2$lambda, type="class")
#confusionMatrix(data = pred_Lasso_1 , reference = yTest)

#pred_Lasso_2 <- predict(lasso2_linkage2, newx=newX , s = lasso2_linkage2$lambda, type="class")
#confusionMatrix(data = pred_Lasso_2 , reference = yTest)

print("LibLineaR predictions...")
# LibLineaR predictions
predictions <- predict(libLinModel, as.matrix(xTest))
confusionMatrix(data = predictions$class, reference = yTest)

print("LDA predictions...")
# Predicting LDA
pred_lda_1 <- predict(ldamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_lda_1$class, reference = yTest)

print("QDA predictions...")
# Predicting QDA
pred_qda_1 <- predict(qdamod_linkage2, newdata = xTest)
confusionMatrix(data = pred_qda_1$class, reference = yTest)

print("Random Forest predictions...")
# Predicting Random Forrest
pred_RF<- predict(train2RF, xTest)
confusionMatrix(data = pred_RF, reference = yTest)

print("Bagging predictions...")
# Bagging predictions
pred_bag <- predict(linkage2Bag, xTest)
confusionMatrix(data = pred_bag, reference = yTest)

print("Gradient Boosting predictions...")
# Gradient Boosting predictions
pred_boost <- round(predict(grBoostFit_linkage2, xTest, type= "prob"))[,2]
confusionMatrix(data = pred_boost, reference = yTest)
