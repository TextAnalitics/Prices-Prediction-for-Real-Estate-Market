#install.packages("caret")
#install.packages("rlang")
#install.packages("dplyr")
#install.packages("caret", dependencies=c("Depends", "Suggests"))
#install.packages("caret", repos = "cran.r-project.org", dependencies = c("Depends", "Imports", "Suggests"))
library(rlang)
library(caret)
updateR()
library(dplyr)
library(tidyverse)

# set pseudorandom number generator
set.seed(10)

# Attach Packages
library(tidyverse)    # data manipulation and visualization
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology
library(RColorBrewer) # customized coloring of plots
setwd('C:\\Users\\RESEARCH\\')
student_data1 <- read.csv("Dataset.csv",
         header = TRUE,        #are there column names in 1st row?
         sep = ",",            #what separates rows?
         strip.white = TRUE,   #strip out extra white space in strings.
         fill = TRUE,          #fill in rows that have unequal numbers of columns
         comment.char = "#",   #character used for comments that should not be read in
         stringsAsFactors = FALSE #Another control for deciding whether characters should be converted to factor
)
student_data1
dim(student_data1)
student_data<-student_data1[,-1]
head(student_data)
dim(student_data)

student_data$HPI.categories <- as.factor(student_data$HPI.categories)
student_data$mieszkania.na.sprzedaÅ..olx <- as.factor(student_data$mieszkania.na.sprzedaÅ..olx)
student_data$mieszkanie.na.sprzedaÅ. <- as.factor(student_data$mieszkanie.na.sprzedaÅ.)
student_data$mieszkania.na.sprzedaz.olx <- as.factor(student_data$mieszkania.na.sprzedaz.olx)
student_data$domy.na.sprzedaz.olx <- as.factor(student_data$domy.na.sprzedaz.olx)
student_data$dom.na.sprzedaz <- as.factor(student_data$dom.na.sprzedaz)
student_data$domy.na.sprzedaÅ.  <- as.factor(student_data$domy.na.sprzedaÅ.)
#student_data$kredyt.na.dom <- as.factor(student_data$kredyt.na.dom)

dim(student_data)

ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkania.na.sprzedaÅ..olx)) + geom_boxplot()
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkanie.na.sprzedaÅ.)) + geom_boxplot()
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkania.na.sprzedaz.olx)) + geom_boxplot()
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = domy.na.sprzedaz.olx)) + geom_boxplot()
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = dom.na.sprzedaz)) + geom_boxplot()
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = domy.na.sprzedaÅ.)) + geom_boxplot()


ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkania.na.sprzedaÅ..olx)) + geom_jitter(aes(colour = HPI.categories))
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkanie.na.sprzedaÅ.)) + geom_jitter(aes(colour = HPI.categories))
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = mieszkania.na.sprzedaz.olx)) + geom_jitter(aes(colour = HPI.categories))
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = domy.na.sprzedaz.olx)) + geom_jitter(aes(colour = HPI.categories))
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = dom.na.sprzedaz)) + geom_jitter(aes(colour = HPI.categories))
ggplot(data = student_data, mapping = aes(x = HPI.categories, y = domy.na.sprzedaÅ.)) + geom_jitter(aes(colour = HPI.categories))

set.seed(2009676)
studentTrain <- createDataPartition(student_data$HPI.categories, list = FALSE, times = 1, p=0.8)
studentTrain
dim(studentTrain)
student_data <- student_data[studentTrain,]
student_data
studentTest <- student_data[-studentTrain,]
studentTest
dim(studentTest)
dim(student_data)
write.csv(student_data,'Train.csv', row.names=FALSE)
write.csv(studentTest,'Test.csv', row.names=FALSE)

# prediction

studentModel <- train(HPI.categories ~ ., data=student_data, method = "knn")
studentModel
studentTestPred <- predict(studentModel, studentTest)
confusionMatrix(studentTestPred, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred, studentTest$HPI.categories)

xtab = table(studentTestPred, studentTest$HPI)
print(xtab)

accuracy = sum(studentTestPred == studentTest$HPI)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))

cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))

ctrl <- trainControl(method = "cv", repeats = 3)
kNNFit2 <- train(HPI.categories~.,
                 data = student_data,
                 method = "knn",
                 tuneLength = 15,
                 trControl = ctrl,
                 preProc = c("center", "scale"))
print(kNNFit2)
plot(kNNFit2)

# load libraries
library(caret)
library(mlbench)
# load the dataset
# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)
fit <- train(HPI.categories~., data=student_data, method="glm", metric="Accuracy", trControl=control)
# display results
print(fit)

#__________________________________________

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Non-Linear methods: Neural Network, SVM, kNN
#Trees and Rules: CART
#Ensembles of Trees: Bagged CART, Random Forest

#________________________________________
#seed=678765689
seed=9676
# Run algorithms using k-fold cross validation
control <- trainControl(method="cv", number=5)
# Run algorithms using Bootstrap resampling
control <- trainControl(method="boot", number=100)
# Run algorithms using repeated k-fold Cross Validation
control <- trainControl(method="repeatedcv", number=5, repeats=3)
# Run algorithms using Leave One Out Cross Validation
control <- trainControl(method="LOOCV")

# GLMNET

set.seed(seed)
fit.glmnet <- train(HPI.categories~., data=student_data, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
fit.glmnet
print(fit.glmnet)
fit.glmnet$bestTune

# SVM Radial
set.seed(seed)
fit.svmRadial <- train(HPI.categories~., data=student_data, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
fit.svmRadial
print(fit.svmRadial)

# kNN
set.seed(seed)
fit.knn <- train(HPI.categories~., data=student_data, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
fit.knn
print(fit.knn)

# Naive Bayes
set.seed(seed)
fit.nb <- train(HPI.categories~., data=student_data, method="nb", metric=metric, trControl=control)
fit.nb
print(fit.nb)

# C5.0
set.seed(seed)
library(C50)
fit.c50 <- train(HPI.categories~., student_data, method="C5.0", metric=metric, trControl=control)
fit.c50
print(fit.c50)

# CART
set.seed(seed)
fit.cart <- train(HPI.categories~., data=student_data, method="rpart", metric=metric, trControl=control)
fit.cart
print(fit.cart)

# Bagged CART
set.seed(seed)
fit.treebag <- train(HPI.categories~., data=student_data, method="treebag", metric=metric, trControl=control)
fit.treebag
print(fit.treebag)

# Random Forest
set.seed(seed)
fit.rf <- train(HPI.categories~., data=student_data, method="rf", metric=metric, trControl=control)
fit.rf
print(fit.rf)
#________________________________________

results <- resamples(list(GLMNET=fit.glmnet, SVM=fit.svmRadial, KNN=fit.knn, CART=fit.cart, Naive_Bayes=fit.nb,C5.0=fit.c50,
                          Bagged_CART=fit.treebag, Random_Forest=fit.rf))
results
# Table comparison
summary(results)

# boxplot comparison
bwplot(results)
# Dot-plot comparison
#dotplot(results)

#_________________________________________________
# GLMNET
studentTestPred_glmnet <- predict(fit.glmnet, studentTest)
confusionMatrix(studentTestPred_glmnet, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_glmnet, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_glmnet, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_glmnet == studentTest$HPI.categories)/length(studentTest$HPI)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab1 = data.frame(accuracy, precision,recall,f )
xtab1

# SVM Radial
studentTestPred_svmRadial <- predict(fit.svmRadial, studentTest)
confusionMatrix(studentTestPred_svmRadial, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_svmRadial, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_svmRadial, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_svmRadial == studentTest$HPI)/length(studentTest$HPI)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab2 = data.frame(accuracy, precision,recall,f )
xtab2

# kNN
studentTestPred_knn <- predict(fit.knn, studentTest)
confusionMatrix(studentTestPred_knn, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_knn, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_knn, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_knn == studentTest$HPI.categories)/length(studentTest$HPI)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab3 = data.frame(accuracy, precision,recall,f )
xtab3

# Bagged CART
studentTestPred_treebag <- predict(fit.treebag, studentTest)
confusionMatrix(studentTestPred_treebag, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_treebag, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_treebag, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_treebag == studentTest$HPI.categories)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab4 = data.frame(accuracy, precision,recall,f )
xtab4

# Random Forest
studentTestPred_rf <- predict(fit.rf, studentTest)
confusionMatrix(studentTestPred_rf, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_rf, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_rf, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_rf == studentTest$HPI.categories)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab5 = data.frame(accuracy, precision,recall,f )
xtab5

# CART
studentTestPred_cart <- predict(fit.cart, studentTest)
confusionMatrix(studentTestPred_cart, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_cart, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_cart, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_cart == studentTest$HPI.categories)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab6 = data.frame(accuracy, precision,recall,f )
xtab6

# Naive Bayes
studentTestPred_nb <- predict(fit.nb, studentTest)
confusionMatrix(studentTestPred_nb, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_nb, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_nb, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_nb == studentTest$HPI.categories)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab7 = data.frame(accuracy, precision,recall,f )
xtab7

# C5.0
studentTestPred_c50 <- predict(fit.c50, studentTest)
confusionMatrix(studentTestPred_c50, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_c50, studentTest$HPI.categories, mode="everything")

xtab = table(studentTestPred_c50, studentTest$HPI.categories)
print(xtab)
accuracy = sum(studentTestPred_c50 == studentTest$HPI.categories)/length(studentTest$HPI.categories)
precision = xtab[1,1]/sum(xtab[,1])
recall = xtab[1,1]/sum(xtab[1,])
f = 2 * (precision * recall) / (precision + recall)
cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
xtab8 = data.frame(accuracy, precision,recall,f )
xtab8
# GLMNET
xtab1
# SVM Radial
xtab2
# kNN
xtab3
# Bagged CART
xtab4
# Random Forest
xtab5
# CART
xtab6
# Naive Bayes
xtab7
# C5.0
xtab8
#_________________________________________________________________________________________
# GLMNET
studentTestPred_glmnet <- predict(fit.glmnet, studentTest)
confusionMatrix(studentTestPred_glmnet, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_glmnet, studentTest$HPI.categories, mode="everything")

# C5.0
studentTestPred_c50 <- predict(fit.c50, studentTest)
confusionMatrix(studentTestPred_c50, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_c50, studentTest$HPI.categories, mode="everything")

# Bagged CART
studentTestPred_treebag <- predict(fit.treebag, studentTest)
confusionMatrix(studentTestPred_treebag, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_treebag, studentTest$HPI.categories, mode="everything")

# Random Forest
studentTestPred_rf <- predict(fit.rf, studentTest)
confusionMatrix(studentTestPred_rf, studentTest$HPI.categories)$overall['Accuracy']
confusionMatrix(studentTestPred_rf, studentTest$HPI.categories, mode="everything")


