library(glmnet)
library(class)

# Step 1: Set random seed and shuffle rows
set.seed(2017)
shuffle <- sample(nrow(predicted1_lda))

predicted1 <- predicted1_lda[shuffle,]
labeled1 <- labeled1_lda[shuffle]

# Knn classification
kf <- list()
err <- NA
for(i in 1:100){ 
  kf[[i]] <- knn.cv(as.matrix(predicted1), labeled1, k=i)
  err[i] <- (nrow(predicted1)-sum(diag(table(labeled1, kf[[i]]))))/nrow(predicted1)
}
plot(err[1:40], type="l", lwd=3, col="red")
which.min(err)
table(kf[[13]], labeled1)

# Step 2: Create 10 equally size folds
folds_svm <- cut(seq(1,nrow(predicted1)),breaks=10,labels=FALSE)

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
  cvData_train <- predicted1[-testIndexes,]
  cvLabels_train <- labeled1[-testIndexes]
  
  cvData_test <- predicted1[testIndexes,]
  cvLabels_test <- labeled1[testIndexes]
  
  # k = 3 for kNN running on predicted1
  knnModel <- knn(train = cvData_train, test = cvData_test, cl = cvLabels_train, k = 13)
  confMatr <- table(knnModel, cvLabels_test)
  
  # predicted on the whole data set
  cvFold_predictions_svm[ind:(ind-1+length(testIndexes))] <- knnModel
  ind <- ind + length(testIndexes)
  
  pred_precision_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
  pred_recall_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
  errorRate_svm[i] <- (nrow(cvData_test)-sum(diag(table(knnModel, cvLabels_test))))/nrow(cvData_test)
  
  # print info fir each iteration
  cat("Fold ", i , " Error rate: ", errorRate_svm[i], "\n")
  cat("Fold ", i , " Precision rate: ", pred_precision_svm[i], "\n")
  cat("Fold ", i , " Recall rate: ", pred_recall_svm[i], "\n")
  
}

avgErrorRate_svm <- mean(errorRate_svm)
avgPrecision_svm <- mean(pred_precision_svm)
avgRecall_svm <- mean(pred_recall_svm)

avgErrorRate_svm
avgPrecision_svm
avgRecall_svm

svmPred_cv <- factor(cvFold_predictions_svm) # classes were 1 and 2, so subtract 1 to have the usual 0 and 1 classes

print("Prediction for left out subsets inside k-fold CV for SVM: ")
table(svmPred_cv, labeled1)
