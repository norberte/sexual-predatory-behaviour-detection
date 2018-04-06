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

# second level cross validation
# LDA cross validated ----------------------------------------------------------
lda_firstLevel_CV <- lda(formula = factor(labels)~., data = data, CV = TRUE)
table(lda_firstLevel_CV$class, labels)

index2 <- which(lda_firstLevel_CV$class == 1)
predicted1_ldaCV <- data[index2,]
labeled1_ldaCV <- labels[index2]




# LDA fitted on the whole data set
lda_firstLevel_classifier <- lda(formula = factor(labels)~., data = data)

# LDA predicted on the whole data set
pred_100k_lda <- predict(lda_firstLevel_classifier, newdata = data)
confusionMatrix(data = pred_100k_lda$class, reference = labels)

#indexing the observations labeled as 1 (predator)
index <- which(pred_100k_lda$class == 1)
predicted1_lda <- data[index,]
labeled1_lda <- labels[index]


# second level cross validation 
# TO BE DONE ...

# 2. CV LibLinear SVM  -----------------------
libLinModel_CV <- LiblineaR(data = predicted1_ldaCV, target = factor(labeled1_ldaCV), cross = 10, type = 3, verbose = TRUE)

# Manual Cross validation 
# Set random seed and optionally shuffle rows
set.seed(2017)

# Step 2: Create 10 equally size folds
folds <- cut(seq(1,nrow(predicted1_ldaCV)),breaks=10,labels=FALSE)

# Step 3: set up array to store MSE results
errorRate <- NA

# Step 4: Perform 10 fold cross validation
for(i in 1:10){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  cvData_train <- predicted1_ldaCV[-testIndexes,]
  cvLabels_train <- labeled1_ldaCV[-testIndexes]
  
  cvData_test <- predicted1_ldaCV[testIndexes,]
  cvLabels_test <- labeled1_ldaCV[testIndexes]
  
  libLinear <- LiblineaR(data = cvData_train, target = factor(cvLabels_train), type = 3)
  predictLibLinear <- predict(libLinear, as.matrix(cvData_test))
  
  errorRate[i] <- (nrow(cvData_test)-sum(diag(table(cvLabels_test, predictLibLinear$predictions))))/nrow(cvData_test)
  cat("Fold ", i , " Error rate: ", errorRate[i], "\n")
}

avgErrorRate <- mean(errorRate)


# fit LiblineaR
libLinModel <- LiblineaR(data = predicted1_ldaCV, target = factor(labeled1_ldaCV), type = 3, verbose = TRUE)

# LibLineaR predictions
predictions <- predict(libLinModel, as.matrix(predicted1_ldaCV))
confusionMatrix(data = predictions$predictions, reference = labeled1_ldaCV)

newPred_withSVM <- lda_firstLevel_CV$class

for(i in 1:length(index2)){
  if(predictions$predictions[i] == 0){
    newPred_withSVM[index2[i]] = 0
  }
}

confusionMatrix(data = newPred_withSVM, reference = labels)




#11. Adabag -----------------------------
library(adabag)
boosting_df <- data.frame(predicted1_ldaCV, Y = factor(labeled1_ldaCV))
boosting_CV <- boosting.cv(formula = Y~., data=boosting_df, v = 10, boos = TRUE, mfinal = 300)

table(boosting_CV$class, labeled1_ldaCV)

newPred_withBoosting <- lda_firstLevel_CV$class

for(i in 1:length(index2)){
  if(boosting_CV$class[i] == 0){
    newPred_withBoosting[index2[i]] = 0
  }
}

confusionMatrix(data = newPred_withBoosting, reference = labels)























# -------------- LDA Method = "mle" Manual 10-fold Cross validation ---------------------------------
# Set random seed and optionally shuffle rows
set.seed(2017)

# Step 2: Create 10 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)

# Step 3: set up array to store error rate results
errorRate2 <- NA
pred_recall2 <- NA
pred_precision2 <- NA
cvFold_predictions2 <- NA

# Step 4: Perform 10 fold cross validation
print("First Level Classifier - LDA MLE: 10 fold cross validation")
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
  ldaModel <- lda(formula = factor(cvLabels_train)~., data = cvData_train, method = "mle") 
  
  # predicted on the whole data set
  pred_lda <- predict(ldaModel, newdata = cvData_test)
  cvFold_predictions2[(((i-1)*10000) + 1):(i*10000)] <- pred_lda$class
  
  # calculate error rate
  errorRate2[i] <- (nrow(cvData_test)-sum(diag(table(pred_lda$class, cvLabels_test))))/nrow(cvData_test)
  confMatr <- table(pred_lda$class, cvLabels_test)
  # Precision for predator = TN / (TN+FN)
  # Recall for predator = TN / (TN+FP)
  pred_precision2[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
  pred_recall2[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
  
  # print info fir eeac iteration
  cat("Fold ", i , " Error rate: ", errorRate2[i], "\n")
  cat("Fold ", i , " Precision rate: ", pred_precision2[i], "\n")
  cat("Fold ", i , " Recall rate: ", pred_recall2[i], "\n")
}

avgErrorRate_ldaMLE <- mean(errorRate2)
avgErrorRate_ldaMLE
avgPrecision_ldaMLE <- mean(pred_precision2)
avgPrecision_ldaMLE
avgRecall_ldaMLE <- mean(pred_recall2)
avgRecall_ldaMLE

ldaMLE_Pred_cv <- factor(cvFold_predictions2 - 1) # classes were 1 and 2, so subtract 1 to have the usual 0 and 1 classes

print("Prediction for left out subsets inside k-fold CV for LDA using method = mle :")
table(ldaMLE_Pred_cv, labels)



