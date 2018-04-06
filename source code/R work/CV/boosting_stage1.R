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

# -------------- Boosting 10-fold Cross validation 
data <- data.frame(data, Y = factor(labels))
print("Second Level Classifier - Version A - Boosting: 10 fold cross validation")
boosting_CV <- boosting.cv(formula = Y~., data=data, v = 10, boos = TRUE)

print("Prediction for left out subsets inside k-fold CV for Boosting:")
conf <- boosting_CV$confusion
conf
boosting_class <- boosting_CV$class