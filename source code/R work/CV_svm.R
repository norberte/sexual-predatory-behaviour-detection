library(LiblineaR)
library(readr)
library('e1071')
library(caret)

set.seed(2017)
samp <- sample(1:154677, 100000)

#import data
labels <- read.csv("C:/Users/Norbert/Desktop/doc2vec/doc2vec_labels.csv", header = TRUE)[samp,]
data <- read.csv("C:/Users/Norbert/Desktop/doc2vec/doc2vec_dim400.csv", header = TRUE)[samp,]

train <-sample(1:100000, 60000)

xTrain <- data[train,]
xTest <- data[-train,]
yTrain <- labels[train]
yTest <- labels[-train]

#result <- tune.svm(x = xTrain, y = factor(yTrain), gamma = c(0.1, 0.25, .5, 1, 2),
#         cost = 10^seq(1, -2, length=20), class.weights = c(.7, .8, .9, .95, .97) )
#saveRDS(result, "word2vec_dim200_maxDoc_svm_tune.rds")

svm_grid <- expand.grid(cost = 10^seq(1, -2, length=20), 
                        Loss = c(0.01, 0.05, 0.1, 0.25, 0.5, 1))

svm_ctrl <- trainControl(method = "cv", number = 10)

svm_fit <- train(x = xTrain, y = factor(yTrain),
                 method = 'svmLinear3',
                 tuneGrid = svm_grid,
                 trControl = svm_ctrl)
saveRDS(svm_fit, "C:/Users/Norbert/Desktop/doc2vec/doc2vec_dim400_svm_CV.rds")

#svm_fit <- readRDS("C:/Users/Norbert/Desktop/doc2vec/doc2vec_dim400_svm_CV.rds")
svm_fit

trellis.par.set(caretTheme())
plot(svm_fit, scales = list(x = list(log = 2)))

pred_classes <- predict(svm_fit, newdata = xTest)
confusionMatrix(data = pred_classes, reference = yTest)


glmnet_fit <- readRDS("C:/Users/Norbert/Desktop/Word2Vec/word2vec_dim200_maxDoc_glmnet_CV.rds")
glmnet_fit

trellis.par.set(caretTheme())
plot(glmnet_fit, scales = list(x = list(log = 2)))

pred_classes <- predict(glmnet_fit, newdata = xTest)
confusionMatrix(data = pred_classes, reference = yTest)
