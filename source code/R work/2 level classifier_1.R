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
lda_simple <- lda(factor(yTrain)~., data = xTrain)

# Predicting LDA
pred_lda_1 <- predict(lda_simple, newdata = xTest)
confusionMatrix(data = pred_lda_1$class, reference = yTest)

index <- which(pred_lda_1$class == 1)

predicted1 <- xTest[index,]
labeled1 <- yTest[index]

# second level classification -----------------------------------------------------------

#import data -----------
newLabels <- read.csv("C:/Users/Norbert/Desktop/TWE output/new_labels.csv", header = TRUE)[-samp,]
newData <- read.csv("C:/Users/Norbert/Desktop/TWE output/minMaxDoc_dim400.csv", header = TRUE)[-samp,]

set.seed(2018)

predatorIndeces <- which(newLabels == 1)
nonPredatorIndeces <- which(newLabels == 0)

predatorLabels <- newLabels[predatorIndeces]
predatorData <- newData[predatorIndeces,]

nonPredatorSample <- sample(nonPredatorIndeces, 284)
nonPredatorLabels <- newLabels[nonPredatorSample]
nonPredatorData <- newData[nonPredatorSample,]

trainSample_secondLayer <- sample(1:284, 250)

xTrain_secondLayer <- rbind(predatorData[trainSample_secondLayer,], nonPredatorData[trainSample_secondLayer,])
yTrain_secondLayer <- NULL
yTrain_secondLayer[1:250] <- predatorLabels[trainSample_secondLayer]
yTrain_secondLayer[251:500] <- nonPredatorLabels[trainSample_secondLayer]

xTest_secondLayer <- rbind(predatorData[-trainSample_secondLayer,], nonPredatorData[-trainSample_secondLayer,])
yTest_secondLayer <- NULL
yTest_secondLayer[1:34] <- predatorLabels[-trainSample_secondLayer]
yTest_secondLayer[35:68] <- nonPredatorLabels[-trainSample_secondLayer]

trainShuffle <- sample(nrow(xTrain_secondLayer))
testShuffle <- sample(nrow(xTest_secondLayer))

xTrain_second <- xTrain_secondLayer[trainShuffle,]
yTrain_second <- yTrain_secondLayer[trainShuffle]
xTest_second <- xTest_secondLayer[testShuffle,]
yTest_second <- yTest_secondLayer[testShuffle]


# 1. Random forrest ----------------------------------------------------------
train2RF <- randomForest(factor(yTrain_secondLayer)~., data = xTrain_secondLayer, importance=TRUE)
train2RF

newPred <- pred_lda_1$class
for(i in 1:length(index)){
  if(train2RF$predicted[i] == 0){
    newPred[index[i]] = 0
  }
}

confusionMatrix(data = newPred, reference = yTest)


# 2. LDA ---------------------------------
L2_lda <- lda(factor(yTrain_secondLayer)~., data = xTrain_secondLayer)

predictions <- predict(L2_lda, newdata = predicted1)
confusionMatrix(data = predictions$class, reference = as.matrix(labeled1))

newPred <- pred_lda_1$class
for(i in 1:length(index)){
  if(predictions$class[i] == 0){
    newPred[index[i]] = 0
  }
}

confusionMatrix(data = newPred, reference = yTest)

# 3 Gradient Boosting ---------------
library(caret)
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 500,shrinkage = 0.1,n.minobsinnode = 10)
grBoostFit <- train(factor(yTrain_secondLayer)~., data = as.matrix(xTrain_secondLayer), method = "gbm", trControl = fitControl,verbose = FALSE, tuneGrid = tune_Grid)

pred_boost <- round(predict(grBoostFit, predicted1, type= "prob"))[,2]
confusionMatrix(data = pred_boost, reference = labeled1)


# LibLineaR predictions
predictions <- predict(libLinModel, as.matrix(predicted1))
confusionMatrix(data = predictions$predictions, reference = as.matrix(labeled1))

newPred <- pred_lda_1$class
for(i in 1:length(index)){
  if(predictions$predictions[i] == 0){
    newPred[index[i]] = 0
  }
}

confusionMatrix(data = newPred, reference = yTest)

# tree based classification
# TO DO

# teigen classification
library(teigen)
teigen_data = rbind(xTrain_secondLayer[1:5000,], predicted1)
teigen_labels = NULL
teigen_labels =  yTrain_secondLayer[1:5000]
teigen_labels[5001:5383] = NA

teigenClass <- teigen(x = teigen_data, models="all", init = "uniform", known = teigen_labels)

#Classification with the iris data set via percentage of data taken to have known membership
library(mmtfa)
mmtfa_2level <- mmtfa(teigen_data,Gs = 2, Qs=1:2, models="all", init="uniform", clas = 93, known=teigen_labels)
mmtfa_2level$tab


