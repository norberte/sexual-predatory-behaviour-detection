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
data <- read.csv("C:/Users/Norbert/Desktop/TWE output/minDoc_dim400.csv", header = TRUE)[samp,]

train <-sample(1:100000, 60000)

xTrain <- data[train,]
xTest <- data[-train,]
yTrain <- labels[train]
yTest <- labels[-train]


#cv_splits <- createFolds(labels, k = 10, returnTrain = TRUE)
#str(cv_splits)


# using caret to cross validate
glmnet_grid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                           lambda = seq(.01, .2, length = 20))
glmnet_ctrl <- trainControl(method = "cv", number = 10)
glmnet_fit <- train(x = xTrain, y = factor(yTrain),
                    method = "glmnet",
                    preProcess = c("center"),
                    tuneGrid = glmnet_grid,
                    trControl = glmnet_ctrl)
saveRDS(glmnet_fit, "C:/Users/Norbert/Desktop/Word2Vec/word2vec_dim400_minDoc_glmnet_CV.rds")
glmnet_fit

trellis.par.set(caretTheme())
plot(glmnet_fit, scales = list(x = list(log = 2)))

pred_classes <- predict(glmnet_fit, newdata = xTest)
confusionMatrix(data = pred_classes, reference = yTest)