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


# LDA first level classification----------------------------------------------------------
lda_firstLevel <- lda(factor(yTrain)~., data = xTrain)

# Predicting LDA for test set (40k set) -----------
pred_40k_lda <- predict(lda_firstLevel, newdata = xTest)
confusionMatrix(data = pred_40k_lda$class, reference = yTest)

#indexing the observations labeled as 1 (predator)
index <- which(pred_40k_lda$class == 1)
predicted1_test <- xTest[index,]
labeled1_test <- yTest[index]




# Predicting LDA for train set (60k set) -----------
pred_60k_lda <- predict(lda_firstLevel, newdata = xTrain)
confusionMatrix(data = pred_60k_lda$class, reference = yTrain)

#indexing the observations labeled as 1 (predator)
index <- which(pred_60k_lda$class == 1)
predicted1_train <- xTrain[index,]
labeled1_train <- yTrain[index]


# second level classification -----------------------------------------------------------

# 1. Random forrest ----------------------------------------------------------
train2RF <- randomForest(factor(labeled1_train)~., data = predicted1_train, importance=TRUE)
train2RF

predictions <- predict(train2RF, as.matrix(predicted1_test))
confusionMatrix(data = predictions, reference = labeled1_test)

newPred <- pred_lda_1$class
for(i in 1:length(index)){
  if(train2RF$predicted[i] == 0){
    newPred[index[i]] = 0
  }
}

confusionMatrix(data = newPred, reference = yTest)

# Bagging
bagging <- randomForest(factor(labeled1_train)~., data = predicted1_train, mtry = 800, importance=TRUE)
bagging

predictions <- predict(bagging, as.matrix(predicted1_test))
confusionMatrix(data = predictions, reference = labeled1_test)


# 2. LibLinear SVM L2-loss -----------------------
library('e1071')
svm_tune <- tune(svm, train.y = factor(labeled1_train), train.x = predicted1_train, kernel="linear", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)

svm_radial <- svm(factor(yTrain)~ ., data = xTrain , kernel='radial', cost=1, gamma = 0.5, scale=FALSE) # cross validated parameters
summary(svm_radial)

pred_svm1 <- predict(svm_radial, predicted1_test)
confusionMatrix(data = pred_svm1 , reference = labeled1_test)

svm_linear <- svm(factor(yTrain)~ ., data = xTrain , kernel='linear', cost=0.1, gamma = 0.5, scale=FALSE) # cross validated parameters
summary(svm_linear)

pred_svm2 <- predict(svm_linear, predicted1_test)
confusionMatrix(data = pred_svm2 , reference = labeled1_test)


libLinModel <- LiblineaR(data = predicted1_train, target = factor(labeled1_train), type = 7, verbose = TRUE)

# LibLineaR predictions
predictions <- predict(libLinModel, as.matrix(predicted1_test))
confusionMatrix(data = predictions$predictions, reference = labeled1_test)

newPred <- pred_lda_1$class
for(i in 1:length(index)){
  if(predictions$predictions[i] == 0){
    newPred[index[i]] = 0
  }
}

confusionMatrix(data = newPred, reference = yTest)

# 3 Gradient Boosting ---------------
library(caret)

fitControl <- trainControl(method = "cv",number = 5)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 300, shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit <- train(Y~., data=train_df, method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)
pred_boost <- round(predict(grBoostFit, predicted1_test, type= "prob"))[,2]
confusionMatrix(data = pred_boost, reference = labeled1_test)

library(gbm)
grBoostFit<- gbm(factor(labeled1_train)~., data = predicted1_train ,distribution = "tdist", n.trees = 500 ,shrinkage = 0.01,cv.folds = 10, interaction.depth = 2)
best.iter <- gbm.perf(grBoostFit, method = "cv")
pred_boost <- round(predict(grBoostFit, predicted1_test, best.iter))-1
confusionMatrix(data = pred_boost, reference = labeled1_test)

# 4. Lasso ----------------------------------------------------------
library(glmnet)
grid <- 10^seq(1, -5, length=100)
grid <- 10^seq(0, -3, length=200)

lasso_cv <- cv.glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha = 1,lambda = grid, family = "binomial")
plot(lasso_cv$glmnet.fit, label=TRUE)
plot(lasso_cv)
lam <- lasso_cv$lambda.min
lamsm <- lasso_cv$lambda.1se
finmodel <- glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha=1, lambda=lam, family = "binomial")
finmodel1 <- glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha=1, lambda=lamsm, family = "binomial")

newX <- as.matrix(as.matrix(predicted1_test))

pred_Lasso_1 <- predict(finmodel, newx=newX , s = finmodel$lambda, type="class")
confusionMatrix(data = pred_Lasso_1, reference = labeled1_test)

pred_Lasso_2 <- predict(finmodel1, newx=newX , s = finmodel1$lambda, type="class")
confusionMatrix(data = pred_Lasso_2, reference = labeled1_test)


# 4.5 Ridge ----------------------------------------------------------
# Ridge ------------------------------------
grid <- 10^seq(2, -2, length=200)
library(MASS)
ridge.final <- lm.ridge(factor(labeled1_train)~. ,data = predicted1_train, lambda=grid)
ridge.final
plot(ridge.final)

# select parameter by minimum GCV
plot(ridge.final$GCV)

# Predict is not implemented so we need to do it ourselves
y.pred.ridge = scale(testRidge[,-1],center = F, scale = ridge.final$scales)%*% ridge.final$coef[,which.min(ridge.final$GCV)] + ridge.final$ym
summary((y.pred.ridge - testRidge[,1])^2)



library(glmnet)
#grid <- 10^seq(3, -6, length=100)
grid <- 10^seq(2, -2, length=200)

ridge_cv <- cv.glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha = 0,lambda = grid, family = "binomial")
plot(ridge_cv$glmnet.fit, label=TRUE)
plot(ridge_cv)
lam <- ridge_cv$lambda.min
lamsm <- ridge_cv$lambda.1se
ridge_finmodel <- glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha=0, lambda=lam, family = "binomial")
ridge_finmodel1 <- glmnet(x = as.matrix(predicted1_train), y = factor(labeled1_train), alpha=0, lambda=lamsm, family = "binomial")

pred_ridge_1 <- predict(ridge_finmodel, newx=as.matrix(predicted1_test) , s = ridge_finmodel$lambda, type="class")
confusionMatrix(data = pred_ridge_1, reference = labeled1_test)

pred_ridge_2 <- predict(ridge_finmodel1, newx=as.matrix(predicted1_test) , s = ridge_finmodel1$lambda, type="class")
confusionMatrix(data = pred_ridge_2, reference = labeled1_test)

# 5. LDA ---------------------------------
lda_moment <- lda(factor(labeled1_train)~., data = predicted1_train)

# Predicting LDA
pred_lda_moment <- predict(lda_moment, newdata = predicted1_test)
confusionMatrix(data = pred_lda_moment$class, reference = labeled1_test)

#6. Knn classification
library(class)
kf <- list()
err <- NA
for(i in 1:100){ 
  kf[[i]] <- knn.cv(as.matrix(predicted1_train), labeled1_train, k=i)
  err[i] <- (569-sum(diag(table(labeled1_train, kf[[i]]))))/569
}
plot(err[1:20], type="l", lwd=3, col="red")
which.min(err)
table(kf[[3]], labeled1_train)

# k = 3 for kNN running on predicted1
knnModel1 <- knn(train = predicted1_train, test = predicted1_test, cl = labeled1_train, k = 3)
table(knnModel1, labeled1_test)

# k = 10 for kNN running on predicted1
knnModel2 <- knn(train = predicted1_train, test = predicted1_test, cl = labeled1_train, k = 10)
table(knnModel2, labeled1_test)

# k = 16 for kNN running on predicted1
knnModel3 <- knn(train = predicted1_train, test = predicted1_test, cl = labeled1_train, k = 16)
table(knnModel3, labeled1_test)

#7. Naive Bayes
library(e1071)
model <- naiveBayes(factor(labeled1_train)~., data=predicted1_train)
prediction <- predict(model, predicted1_test, type = "class")
table(prediction, labeled1_test)

#8. teigen classification
library(teigen)
teigen_data = rbind(xTrain_ldaPredicted1, predicted1)
teigen_labels = NULL
teigen_labels =  yTrain_ldaLabeled1[1:191]
teigen_labels[192:574] = NA

teigenClass <- teigen(x = teigen_data[,201:600], models="all", init = "uniform", gauss = TRUE, known = teigen_labels)
table(teigenClass$classification[192:574], labeled1)

#9. Logistic Regression
logModel <- glm(factor(yTrain_ldaLabeled1)~., family=binomial(link='logit'), data = xTrain_ldaPredicted1)
summary(logModel) # get the summary of the model
results <- predict(logModel,newdata=predicted1,type='response')
testPredictions <- ifelse(results > 0.5,1,0) # get the right predictions
table(testPredictions, labeled1)

# stepAIC model, which gets in another logistic regression model
stepLog <- stepAIC(logModel, direction = "backward", trace = FALSE)
improvedLogModel = glm(stepLog$call$formula, family=binomial(link='logit'), data = xTrain_ldaPredicted1)
summary(improvedLogModel)
results <- predict(improvedLogModel,newdata=test[,-58],type='response')
testPredictions <- ifelse(results > 0.5,1,0)
table(testPredictions, testDecision)
#(12+93)/1000
# Logistic reg. w/ stepAIC backward direction : Misclassification rate = 0.105 (10.5 %)


#10. Adaboost ---------------------------------
library(fastAdaboost)
train_df <- data.frame(predicted1_train, Y = factor(labeled1_train))

adaboost_model <- adaboost(Y~., data=train_df, nIter = 500)
pred <- predict(adaboost_model,newdata=predicted1_test)
table(pred$class,labeled1_test)



real_adaboost <- real_adaboost(Y~., data=train_df, nIter = 500)
pred_real <- predict(real_adaboost,newdata=predicted1_test)
print(paste("Real Adaboost Error on fakedata:", pred_real$error))
table(pred_real$class,labeled1_test)


#11. Adabag -----------------------------
library(adabag)
boosting_CV <- boosting.cv(formula = Y~., data=train_df, v = 5, boos = TRUE, mfinal = 100)
bagging_CV <- bagging.cv(formula = Y~., data=train_df, v = 5, mfinal = 100)

adaboost <- boosting(formula = Y~., data=train_df, boos=TRUE, mfinal = 100)
importanceplot(adaboost)
pred_boosting<- predict.boosting(adaboost, newdata=predicted1_test)
table(pred_boosting$class, labeled1_test)


bagging <- bagging(formula = Y~., data=train_df, mfinal = 100)
pred_bagging<-predict.bagging(bagging, newdata=predicted1_test)
table(pred_bagging$class, labeled1_test)


adaboost_2 <- boosting(formula = Y~., data=train_df, boos=TRUE, mfinal = 500)
importanceplot(adaboost_2)
pred_boosting_2<- predict.boosting(adaboost_2, newdata=predicted1_test)
table(pred_boosting_2$class, labeled1_test)


bagging_2 <- bagging(formula = Y~., data=train_df, mfinal = 300)
pred_bagging_2 <- predict.bagging(bagging_2, newdata=predicted1_test)
table(pred_bagging_2$class, labeled1_test)

