library(LiblineaR)
library(readr)
library('e1071')
library(MASS)
library(caret)
library(gbm)
library(randomForest)

set.seed(2017)
samp <- sample(1:154677, 100000)

#import data
labels <- read.csv("doc2vec_labels.csv", header = TRUE)[samp,]
data <- read.csv("doc2vec_dim400.csv", header = TRUE)[samp,]

train <-sample(1:100000, 60000)

xTrain <- data[train,]
xTest <- data[-train,]
print("imported data ...")
yTrain <- labels[train]
yTest <- labels[-train]
print("imported labels ...")


# LibLinear SVM L2-loss
libLinModel <- LiblineaR(data = as.matrix(xTrain), target = yTrain, type = 1, verbose = TRUE)
saveRDS(libLinModel, "doc2vec400_LiblineaR.rds")
print("LibLineaR finished training...")

# SVM
svm_linear2 <- svm(factor(yTrain)~ ., data = xTrain , kernel='linear', cost=0.1, gamma = 0.5, scale=FALSE)
summary(svm_linear2)
saveRDS(svm_linear2, "doc2vec400_SVM.rds")
print("SVM finished training...")

# LDA ----------------------------------------------------------
ldamod_linkage2 <- lda(factor(yTrain)~., data = xTrain)
saveRDS(ldamod_linkage2, "doc2vec400_LDA.rds")
print("LDA finished training...")

# QDA ----------------------------------------------------------
qdamod_linkage2 <- qda(factor(yTrain)~., data = xTrain)
saveRDS(qdamod_linkage2, "doc2vec400_QDA.rds")
print("QDA finished training...")

# Random forrest ----------------------------------------------------------
train2RF <- randomForest(factor(yTrain)~., data = xTrain, importance=TRUE)
train2RF
saveRDS(train2RF, "doc2vec400_randomForest.rds")
print("Random Forest finished training...")

# Gradient Boosting ---------------
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit_linkage2 <- train(factor(yTrain)~., data = xTrain, method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)
saveRDS(grBoostFit_linkage2, "doc2vec400_grBoosting.rds")
print("Gradient Boosting finished training...")

# bagging ----------------------------------------------------------
linkage2Bag <- randomForest(factor(yTrain)~., data = xTrain, mtry=400, importance=TRUE)
linkage2Bag
saveRDS(linkage2Bag, "doc2vec400_bagging.rds")
print("Bagging finished training...")
