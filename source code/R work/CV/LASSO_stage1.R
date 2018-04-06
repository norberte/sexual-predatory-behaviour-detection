library(LiblineaR)
library(readr)
library('e1071')
library(MASS)
library(caret)
library(adabag)
library(randomForest)
library(glmnet)

#set seed, create sample 
set.seed(2017)
samp <- sample(1:155113, 100000)

# import data
labels <- read.csv("C:/Users/Norbert/Desktop/TWE output/new_labels.csv", header = TRUE)[samp,]
data <- read.csv("C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim400.csv", header = TRUE)[samp,]


# LASSO Manual 10-fold Cross validation
# Step 1: Set random seed and shuffle rows
set.seed(2017)
shuffle <- sample(nrow(data))
labels <- labels[shuffle]
data <- data[shuffle,]

# CV of grid to find proper lambda for model CV
grid <- 10^seq(-1, -5, length=100)

lasso_cv <- cv.glmnet(x = as.matrix(data), y = factor(labels), alpha = 1,lambda = grid, family = "binomial")
plot(lasso_cv)
lamsm <- lasso_cv$lambda.1se
lam <- lasso_cv$lambda.min
lamsm
lam

finmodel <- glmnet(x = as.matrix(data), y = factor(labels), alpha=1, lambda=lam, family = "binomial")
finmodel1 <-glmnet(x = as.matrix(data), y = factor(labels), alpha=1, lambda=lamsm, family = "binomial")

pred_Lasso <- predict(finmodel, newx= as.matrix(data), s = finmodel$lambda, type="class")
confMatr <- table(pred_Lasso, labels)

lasso1_precision<- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
lasso1_recall <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
lasso1_errorRate <- (nrow(data)-sum(diag(table(pred_Lasso, labels))))/nrow(data)


pred_Lasso2 <- predict(finmodel1, newx= as.matrix(data), s = finmodel1$lambda, type="class")
confMatr2 <- table(pred_Lasso2, labels)

lasso2_precision<- confMatr2[2,2] / (confMatr2[2,2] + confMatr2[1,2])
lasso2_recall <- confMatr2[2,2] / (confMatr2[2,2] + confMatr2[2,1])
lasso2_errorRate <- (nrow(data)-sum(diag(table(pred_Lasso2, labels))))/nrow(data)






# Step 2: Create 10 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)

# Step 3: set up array to store error rate results
errorRate <- NA
pred_recall <- NA
pred_precision <- NA
cvFold_predictions <- NA

# Step 4: Perform 10 fold cross validation
print("First Level Classifier - LASSO: 10 fold cross validation")
for(i in 1:10){
  # Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  # inner train sets
  cvData_train <- data[-testIndexes,]
  cvLabels_train <- labels[-testIndexes]
  
  # inner test sets
  cvData_test <- data[testIndexes,]
  cvLabels_test <- labels[testIndexes]
  
  # fit LASSO model
  finmodel <- glmnet(x = as.matrix(cvData_train), y = factor(cvLabels_train), alpha=1, lambda=lam, family = "binomial", maxit = 20000)
  
  # predicted on the whole data set
  pred_Lasso <- predict(finmodel, newx= as.matrix(cvData_test), s = finmodel$lambda, type="class")
  cvFold_predictions[(((i-1)*10000) + 1):(i*10000)] <- pred_Lasso
  
  # calculate error rate
  errorRate[i] <- (nrow(cvData_test)-sum(diag(table(pred_Lasso, cvLabels_test))))/nrow(cvData_test)
  confMatr <- table(pred_Lasso, cvLabels_test)
  
  # Precision for predator = TN / (TN+FN)
  # Recall for predator = TN / (TN+FP)
  pred_precision[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
  pred_recall[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
  
  # print info fir eeac iteration
  cat("Fold ", i , " Error rate: ", errorRate[i], "\n")
  cat("Fold ", i , " Precision rate: ", pred_precision[i], "\n")
  cat("Fold ", i , " Recall rate: ", pred_recall[i], "\n")
}

avgErrorRate_ldaMoment <- mean(errorRate)
avgErrorRate_ldaMoment
avgPrecision_ldaMoment <- mean(pred_precision)
avgPrecision_ldaMoment
avgRecall_ldaMoment <- mean(pred_recall)
avgRecall_ldaMoment

ldaPred_cv <- factor(cvFold_predictions)

print("Prediction for left out subsets inside k-fold CV for LASSO:")
confusionMatrix(data = ldaPred_cv, reference = labels)