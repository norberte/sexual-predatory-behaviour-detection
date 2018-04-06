library(readr)
library(MASS)
library(caret)

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
confusionMatrix(data = ldaPred_cv, reference = labels)

# indexing the observations labeled as 1 (predator)
index <- which(ldaPred_cv == 1)
predicted1_lda <- data[index,]
labeled1_lda <- labels[index]