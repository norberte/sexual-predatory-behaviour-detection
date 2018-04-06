library(LiblineaR)
library(readr)
library('e1071')
library(MASS)
library(caret)
library(gbm)
library(randomForest)

set.seed(2017)
samp <- sample(1:155113, 100000)

#import data
labels <- read.csv("C:/Users/Norbert/Desktop/TWE output/new_labels.csv", header = TRUE)[samp,]
data <- read.csv("C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim400.csv", header = TRUE)[samp,]

train <-sample(1:100000, 60000)

xTrain <- data[train,]
xTest <- data[-train,]
yTrain <- labels[train]
yTest <- labels[-train]

# SVM -------------
svm_linear <- svm(factor(yTrain)~ ., data = xTrain , type = "C-classification", kernel='linear',
                   cost=10, gamma = 0.1, scale=FALSE)
saveRDS(svm_linear, "C:/Users/Norbert/Desktop/Word2Vec/new models/word2vec_minMaxDoc400_SVM_linear.rds")
pred_svm_1 <- predict(svm_linear, xTest)
confusionMatrix(data = pred_svm_1 , reference = yTest)


svm_radial <- svm(factor(yTrain)~ ., data = xTrain , type = "C-classification", kernel='radial',
                  cost=10, gamma = 0.1, scale=FALSE)
saveRDS(svm_radial, "C:/Users/Norbert/Desktop/Word2Vec/new models/word2vec_minMaxDoc400_SVM_radial.rds")
pred_svm_2 <- predict(svm_radial, xTest)
confusionMatrix(data = pred_svm_2 , reference = yTest)


svm_polynomial <- svm(factor(yTrain)~ ., data = xTrain , type = "C-classification", kernel='polynomial',
                  cost=10, gamma = 0.1, scale=FALSE)
saveRDS(svm_polynomial, "C:/Users/Norbert/Desktop/Word2Vec/new models/word2vec_minMaxDoc400_SVM_polynomial.rds")
pred_svm_3 <- predict(svm_polynomial, xTest)
confusionMatrix(data = pred_svm_3 , reference = yTest)


svm_sigmoid <- svm(factor(yTrain)~ ., data = xTrain , type = "C-classification", kernel='sigmoid',
                   cost=10, gamma = 0.1, scale=FALSE)
saveRDS(svm_sigmoid, "C:/Users/Norbert/Desktop/Word2Vec/new models/word2vec_minMaxDoc400_SVM_sigmoid.rds")
pred_svm_4 <- predict(svm_sigmoid, xTest)
confusionMatrix(data = pred_svm_4 , reference = yTest)






# Lasso CV ------------------
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

# Lasso ------------
library(glmnet)
b <- as.matrix(xTrain)
newX <- as.matrix(xTest)
y <- factor(yTrain)
newY <- factor(yTest)

lasso <- glmnet(x = b, y= y, alpha=1, nlambda=100, family = "binomial", standardize = FALSE)

newX <- as.matrix(xTest)

pred_Lasso_1 <- predict(lasso1_linkage2, newx=newX , s = lasso1_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_1 , reference = yTest)

pred_Lasso_2 <- predict(lasso2_linkage2, newx=newX , s = lasso2_linkage2$lambda, type="class")
confusionMatrix(data = pred_Lasso_2 , reference = yTest)

saveRDS(svm_linear, "C:/Users/Norbert/Desktop/Word2Vec/LASSO + LDA + LibLineaR new models/word2vec_minMaxDoc400_SVM_linear.rds")
