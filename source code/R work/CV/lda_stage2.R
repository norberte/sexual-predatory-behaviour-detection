library(glmnet)
# -------------- LibLinear SVM Manual 10-fold cross validation 
# Step 1: Set random seed and shuffle rows
set.seed(2017)
shuffle <- sample(nrow(predicted1_lda))

predicted1 <- predicted1_lda[shuffle,]
labeled1 <- labeled1_lda[shuffle]

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
  
  # CV of grid to find proper lambda for model CV
  grid <- 10^seq(1, -4, length=100)
  
  lasso_cv <- cv.glmnet(x = as.matrix(cvData_train), y = factor(cvLabels_train), alpha = 1,lambda = grid, family = "binomial")
  lam <- lasso_cv$lambda.min
  
  finmodel <- glmnet(x = as.matrix(cvData_train), y = factor(cvLabels_train), alpha=1, lambda=lam, family = "binomial")
  
  pred_Lasso <- predict(finmodel, newx= as.matrix(cvData_test), s = finmodel$lambda, type="class")
  confMatr <- table(pred_Lasso, cvLabels_test)
  
  # predicted on the whole data set
  cvFold_predictions_svm[ind:(ind-1+length(testIndexes))] <- pred_Lasso
  ind <- ind + length(testIndexes)
  
  pred_precision_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[1,2])
  pred_recall_svm[i] <- confMatr[2,2] / (confMatr[2,2] + confMatr[2,1])
  errorRate_svm[i] <- (nrow(cvData_test)-sum(diag(table(pred_Lasso, cvLabels_test))))/nrow(cvData_test)
  
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
