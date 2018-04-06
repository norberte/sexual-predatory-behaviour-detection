library(LiblineaR)
library(readr)
library('e1071')
library(MASS)
library(caret)
library(adabag)

#set seed, create sample 
set.seed(2017)
samp <- sample(1:155113, 100000)

# import data
labels <- read.csv("C:/Users/Norbert/Desktop/TWE output/new_labels.csv", header = TRUE)[samp,]
data <- read.csv("C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim400.csv", header = TRUE)[samp,]


# LDA Method = moment Manual 10-fold Cross validation
# Step 1: Set random seed and shuffle rows
set.seed(2017)
shuffle <- sample(nrow(data))
labels <- labels[shuffle]
data <- data[shuffle,]

# Step 2: Create 10 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)

# Step 3: set up array to store error rate results
errorRate <- NA
pred_recall <- NA
pred_precision <- NA
cvFold_predictions <- NA

# Step 4: Perform 10 fold cross validation
print("First Level Classifier - LDA: 10 fold cross validation")
for(i in 1:10){
  # Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  # inner train sets
  cvData_train <- data[-testIndexes,]
  cvLabels_train <- labels[-testIndexes]
  
  # inner test sets
  cvData_test <- data[testIndexes,]
  cvLabels_test <- labels[testIndexes]
  
  # fit lda model
  ldaModel <- lda(formula = factor(cvLabels_train)~., data = cvData_train) 
  
  # predicted on the whole data set
  pred_lda <- predict(ldaModel, newdata = cvData_test)
  cvFold_predictions[(((i-1)*10000) + 1):(i*10000)] <- pred_lda$class
  
  # calculate error rate
  errorRate[i] <- (nrow(cvData_test)-sum(diag(table(pred_lda$class, cvLabels_test))))/nrow(cvData_test)
  confMatr <- table(pred_lda$class, cvLabels_test)
  
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

ldaPred_cv <- factor(cvFold_predictions - 1) # classes were 1 and 2, so subtract 1 to have the usual 0 and 1 classes

print("Prediction for left out subsets inside k-fold CV for LDA:")
table(ldaPred_cv, labels)

# indexing the observations labeled as 1 (predator)
index <- which(ldaPred_cv == 1)
predicted1_lda <- data[index,]
labeled1_lda <- labels[index]


# -------------- Boosting 10-fold Cross validation 
set.seed(2017)
shuffle <- sample(nrow(predicted1_lda))

predicted1 <- predicted1_lda[shuffle,]
labeled1 <- labeled1_lda[shuffle]

boosting_df <- data.frame(predicted1, Y = factor(labeled1))
boosting_CV <- boosting.cv(formula = Y~., data=boosting_df, v = 10, boos = TRUE, mfinal = 300)

print("Prediction for left out subsets inside k-fold CV for Boosting:")
boosting_CV$confusion

#Mapping back (optional)
finalPred_withBoosting <- ldaPred_cv

for(i in 1:length(index)){
  if(boosting_CV$class[i] == 0){
    finalPred_withBoosting[index[i]] = 0
  }
}

print("Second Level Classifier - Version A - Boosting: Mapping back (optional) ")
confusionMatrix(data = finalPred_withBoosting, reference = labels)


# -------------- LibLinear SVM Manual 10-fold cross validation 
# Step 1: Set random seed and optionally shuffle rows
set.seed(2017)

# Step 2: Create 10 equally size folds
folds_svm <- cut(seq(1,nrow(predicted1_lda)),breaks=10,labels=FALSE)

# Step 3: set up array to store MSE results
errorRate_svm <- NA
pred_recall_svm <- NA
pred_precision_svm <- NA
cvFold_predictions_svm <- NA
ind = 1

# Step 4: Perform 10 fold cross validation
print("Second Level Classifier - Version B - SVM: 10 fold cross validation ")
for(i in 1:10){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds_svm==i,arr.ind=TRUE)
  cvData_train <- predicted1_lda[-testIndexes,]
  cvLabels_train <- labeled1_lda[-testIndexes]
  
  cvData_test <- predicted1_lda[testIndexes,]
  cvLabels_test <- labeled1_lda[testIndexes]
  
  libLinear <- LiblineaR(data = cvData_train, target = factor(cvLabels_train), type = 3)
  predictLibLinear <- predict(libLinear, as.matrix(cvData_test))
   
  # predicted on the whole data set
  cvFold_predictions_svm[ind:(ind-1+length(testIndexes))] <- predictLibLinear$predictions
  ind <- ind + length(testIndexes)
  
  # calculate error rate
  errorRate_svm[i] <- (nrow(cvData_test)-sum(diag(table(cvLabels_test, predictLibLinear$predictions))))/nrow(cvData_test)
  
  # Precision for predator = TN / (TN+FN)
  # Recall for predator = TN / (TN+FP)
  confMatr <- table(predictLibLinear$predictions, cvLabels_test)
  pred_precision_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
  pred_recall_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
  
  # print info fir each iteration
  cat("Fold ", i , " Error rate: ", errorRate_svm[i], "\n")
  cat("Fold ", i , " Precision rate: ", pred_precision_svm[i], "\n")
  cat("Fold ", i , " Recall rate: ", pred_recall_svm[i], "\n")
  
}

avgErrorRate_svm <- mean(errorRate_svm)
avgPrecision_svm <- mean(pred_precision_svm)
avgRecall_svm <- mean(pred_recall_svm)

svmPred_cv <- factor(cvFold_predictions_svm - 1) # classes were 1 and 2, so subtract 1 to have the usual 0 and 1 classes

print("Prediction for left out subsets inside k-fold CV for SVM: ")
table(svmPred_cv, labeled1_lda)

#Mapping back (optional)
finalPred_withSVM <- ldaPred_cv

for(i in 1:length(index)){
  if(svmPred_cv[i] == 0){
    finalPred_withSVM[index[i]] = 0
  }
}

print("Second Level Classifier - Version B - SVM: Mapping back (optional) ")
confusionMatrix(data = finalPred_withSVM, reference = labels)
