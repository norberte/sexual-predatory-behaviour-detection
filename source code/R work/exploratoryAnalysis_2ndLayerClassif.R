library(randomForest)
# LDA first level classification----------------------------------------------------------
lda_simple <- lda(factor(yTrain)~., data = xTrain)

# Predicting LDA
pred_lda_1 <- predict(lda_simple, newdata = xTest)
confusionMatrix(data = pred_lda_1$class, reference = yTest)

index <- which(pred_lda_1$class == 1)

predicted1 <- xTest[index,]
labeled1 <- yTest[index]

# Write dim200 to TSV file
write.table(predicted1_test, file = "C:/Users/Norbert/Desktop/predicted1_test_word2Vec_minMax400.tsv",row.names=FALSE, na="",col.names=FALSE, sep="\t")
write.table(labeled1_test, file = "C:/Users/Norbert/Desktop/labeled1_test_Labels_word2Vec_minMax400.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")

######################## DIM REDUCTION ------------------

library(HSAUR2)

# PCA on all words ---------------------------------------------------------------------
secondLevel_pca <- prcomp(predicted1_train, scale = FALSE, retx = TRUE)
#Scree plot
plot(secondLevel_pca, type="lines")

#Importance of components (variation explained)
summary(secondLevel_pca)

PCA_scores <- round(secondLevel_pca$x[,1:100], 5)



#--------------------------------
prediction_pca <- prcomp(predicted1_test, scale = TRUE, retx = TRUE)
#Scree plot
plot(prediction_pca, type="lines")

#Importance of components (variation explained)
summary(prediction_pca)

predicted1_PCA_scores <- round(prediction_pca$x[,1:300], 5)

#----------------------------------


library(tree)
# creating the classification tree with the scores obtained from PCA
pcaTree <- tree(factor(labeled1_test)~., data = data.frame(predicted1_PCA_scores))
# summary of the classification tree
summary(pcaTree)
plot(pcaTree)
text(pcaTree)

# pruning the pca classification tree
set.seed(1779)
cvspam <- cv.tree(pcaTree, FUN=prune.misclass) # cross valadation
plot(cvspam, type="b")
p.cvspam <- prune.misclass(pcaTree, best=11) # choose to prune it at 2
plot(p.cvspam) # the pruned classification tree's splits
text(p.cvspam)
summary(p.cvspam) # summary of the pruned classification tree


library(adabag)
fake <- data.frame(PCA_scores, Y = factor(labeled1_train))

adaboost2 <- boosting(formula = Y~., data=fake, boos=TRUE)
importanceplot(adaboost2)
predboosting<- predict.boosting(adaboost2, newdata=data.frame(predicted1_PCA_scores))
table(predboosting$class, labeled1_test)




train2RF <- randomForest(factor(labeled1_test)~., data = predicted1_PCA_scores, importance=TRUE)
train2RF

predictions <- predict(train2RF, as.matrix(predicted1_PCA_scores))
confusionMatrix(data = predictions, reference = labeled1_test)



library(psych)
pred1_fa = fa(predicted1_test, nfactors=10, rotate="varimax", SMC=FALSE, fm="minres")
pred1_fa
fa_loading = pred1_fa$loadings
fa_loading


bodfa <- factanal(body[,-25], 2, rotation="varimax")
bodfa



svd_predicted1_train <- svd(predicted1_train)
prop.table(svd_predicted1_train$d^2)


# Exploratory MDS + FA + PCA -------------
# MDS -------------------------------------
mds <- cmdscale(dist(predicted1_test),eig=TRUE, k=300) # k is the number of dim
which(mds$eig > 1)
dimensionsKept <- mds$points[,1:10]

# boosting cv on mds reduced dim
library(adabag)
test_df <- data.frame(dimensionsKept, Y = factor(labeled1_test))
boosting_CV <- boosting.cv(formula = Y~., data=test_df, v = 5, boos = TRUE, mfinal = 500)
boosting_CV

# tsne
library(tsne)
tsne_reducedDim = tsne(dimensionsKept, perplexity=50)
plot(tsne_reducedDim )

tsne_original = tsne(predicted1_test, perplexity=50)
plot(tsne_original)



# Determine Number of Factors to Extract         
library(nFactors)
ev <- eigen(cor(predicted1_test)) # get eigenvalues
ap <- parallel(subject=nrow(predicted1_test),var=ncol(predicted1_test),rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)        # 4 factors to extract  (or 118 based on eigen values)

which(ev$values > 1)



# Principal Axis Factor Analysis-----------------
library(psych)
fit <- factor.pa(predicted1_test, nfactors=4)
fit # print results

fa_loadings <- fit$loadings




# Maximum Likelihood Factor Analysis --------------------
# entering raw data and extracting 33 factors,with varimax rotation 
factor33 <- factanal(predicted1_test, 118, rotation="varimax")   # does not work
print(factor33, digits=2, cutoff=.3, sort=TRUE)

# Exploratory Factor Analysis ---------------------------
corMat <- cor(predicted1_test)
solution <- fa(r = corMat, nfactors = 4, rotate = "oblimin", fm = "pa")
solution

load <- solution$loadings[,c(1,3)] # plot factor 1 by factor 2 
plot(load,type="n") # set up plot 
text(load,labels=labeled1_test,cex=.7) # add variable names




# PCA Variable Factor Map---------------- 
library(FactoMineR)
result <- PCA(predicted1_test) # graphs generated automatically


# Truncated singular value decomposition (SVD)------------------
dat <- as.matrix(predicted1_test)
s <- svd(dat)
plot(cumsum(s$d^2/sum(s$d^2))) # % explained variance

pc.use <- 1
recon <- s$u[,pc.use] %*% diag(s$d[pc.use], length(pc.use), length(pc.use)) %*% t(s$v[,pc.use])
row.names(recon) <- read_lines(file = "C:/Users/Norbert/Desktop/research2017/relationVectors_metadata.tsv") 

linModl <- glm(factor(labeled1_test)~., family=binomial(link='logit'),data=data.frame(recon[,1]))
summary(linModl)

