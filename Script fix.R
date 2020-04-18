#Loading required packages
library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(rpart)
library(randomForest)
library(caret)
library(reshape2)
library(rpart.plot)

bank <- read.csv("Bank Marketing.csv")

head(bank)
library(dplyr)
which(duplicated(bank))

for (var in 1:ncol(bank)) {
  bank[is.na(bank[,var]), var] <- 
    median(as.numeric(bank[,var]), na.rm=TRUE)
}

sapply(bank, function(x) sum(is.na(x)))
summary(bank)
str(bank)

missmap(bank)

#Visualisasi data numerik 
#heatmap 
cor.mat <- round(cor(bank[,c(1,6,10,12,13,14,15)]),2)
melted.cor.mat <- melt(cor.mat) 
ggplot(melted.cor.mat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() + geom_text(aes(x=Var1, y=Var2, label=value))


library(ggplot2)
theme_set(theme_classic())

# Plot
g <- ggplot(bank, aes(y))
g + geom_density(aes(fill=factor(duration)), alpha=0.8) + 
  labs(title="Density plot", 
       subtitle="City Mileage Grouped by Number of cylinders",
       caption="Source: mpg",
       x="Education",
       fill="# Cylinders")

library(ggplot2)
theme_set(theme_classic())

# Plot
g <- ggplot(bank, aes(education, balance))
g + geom_boxplot(varwidth=T, fill="plum") + 
  labs(title="Box plot", 
       subtitle="City Mileage grouped by Class of vehicle",
       caption="Source: mpg",
       x="Education",
       y="Balance") + ylim(0,20000)

library(ggplot2)
theme_set(theme_bw())

# Plot`
library(ggplot2)
library(ggcorrplot)
# Correlation matrix

corr <- round(cor(bank[,c(1,6,10,12,13,14,15)]), 2)

# Plot
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Bank Marketing", 
           ggtheme=theme_bw)

plot(x=bank$contact, y=bank$y, xlab = "contact", ylab="Income >50K Probability", ylim=c(0,1))

#Membagi data train dan test
set.seed(1234)

sample_size <- floor(0.7*nrow(bank))
train_index <- sample(seq_len(nrow(bank)), size=sample_size)
train_data <- bank[train_index,]
test_data <- bank[-train_index,]
x = train_data[,-17]
y = train_data$y

#Evaluation
evaluation <- function(data,accuracy){
  precision <- precision(data)
  recall <- recall(data)
  f <- F_meas(data)
  
  cat(paste("Precision:\t", format(precision, digits=6), "\n",sep=" "))
  cat(paste("Recall:\t\t", format(recall, digits=6), "\n",sep=" "))
  cat(paste("F-measure:\t", format(f, digits=6), "\n",sep=" "))
}

#Decision tree Gini

dtreemodel_rpart_gini <- rpart(y~., train_data, parms = list(split = "gini"))
predict_dtreerpart_gini <- predict(dtreemodel_rpart_gini, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart_gini, test_data$y)
rpart.plot(dtreemodel_rpart_gini, main = "gini Index")#Menampilkan tree
printcp(dtreemodel_rpart_gini) # mengetahui CP dari tree
plotcp(dtreemodel_rpart_gini)# Menampilkan plot XP dan X-Val realtive error
resultdt <- table(predict_dtreerpart_gini, test_data$y)
cmdt <- confusionMatrix(predict_dtreerpart_gini, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

#Prune tree dengan indikator CP GINI
ptree <- prune(dtreemodel_rpart_gini,cp = dtreemodel_rpart_gini$cptable[which.min(dtreemodel_rpart_gini$cptable[,"xerror"]),"CP"])
rpart.plot(ptree, uniform=TRUE, main="Pruned Classification Gini Tree")
printcp(ptree)
plotcp(ptree)
predict_dtreerpart <- predict(ptree, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)


plot(perf, avg= "threshold", colorize=T, lwd= 3,
     main= "... Precision/Recall graphs ...")
plot(perf, lty=3, col="grey78", add=T)


#Decision tree entrophy
dtreemodel_rpart_entro <- rpart(y~., train_data, parms = list(split = "information"))
predict_dtreerpart_entro <- predict(dtreemodel_rpart_entro, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart_entro, test_data$y)
printcp(dtreemodel_rpart_entro)# mengetahui CP dari tree
plotcp(dtreemodel_rpart_entro)# Menampilkan plot XP dan X-Val realtive error
rpart.plot(dtreemodel_rpart_entro, main = "Entrophy")#Menampilkan tree
resultdt <- table(predict_dtreerpart_entro, test_data$y)
cmdt <- confusionMatrix(predict_dtreerpart_entro, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

#Prune tree dengan indikator CP Entrophy
ptree <- prune(dtreemodel_rpart_entro,cp = dtreemodel_rpart_entro$cptable[which.min(dtreemodel_rpart_entro$cptable[,"xerror"]),"CP"])
rpart.plot(ptree, uniform=TRUE, main="Pruned Classification Entrophy Tree")
printcp(ptree)
plotcp(ptree)
predict_dtreerpart <- predict(ptree, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)


#Naive bayes
#Building a model

library(e1071)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
Predict <- predict(model,newdata = test_data )
X <- varImp(model)
plot(X)
#naive bayes yg praktikum
nbmodel <- naiveBayes(y~., data=train_data)
nbpredict <- predict(nbmodel, test_data[,1:16], type="class")

confusionMatrix(nbpredict,test_data$y)

library(caret)
library(klaR)
trainControl <- trainControl(method="cv", number=5)
fit.nb <- train(y~., data=train_data, method="nb",
                metric="Accuracy", trControl=trainControl)

predict.nb <- predict(fit.nb, test_data[,1:16])

confusionMatrix(predict.nb, test_data$class)


#Drop kolom yang tidak signifikan dan membagi kembali data tes dan training
col_to_drop <- c("Ã¯..age","age","month","day","marital","job")
bank_drop <- bank[,!(names(bank) %in% col_to_drop)]

#Misah data train dan test
set.seed(1234)

sample_size <- floor(0.7*nrow(bank_drop))
train_index <- sample(seq_len(nrow(bank_drop)), size=sample_size)
train_data <- bank_drop[train_index,]
test_data <- bank_drop[-train_index,]
x = train_data[,-11]
y = train_data$y


#################################naive bayes yg praktikum no laplace###########################
nbmodel <- naiveBayes(y~., data=train_data)
nbpredict <- predict(nbmodel, test_data[,1:12], type="class")

confusionMatrix(nbpredict,test_data$y)

library(caret)
library(klaR)
trainControl <- trainControl(method="cv", number=5)
fit.nb <- train(y~., data=train_data, method="nb",
                metric="Accuracy", trControl=trainControl)

predict.nb <- predict(fit.nb, test_data[,1:12])

confusionMatrix(predict.nb, test_data$y)
X <- varImp(fit.nb)
plot(X)

###############################naive bayes yg praktikum laplace#####################
nbmodel <- naiveBayes(y~., data=train_data, laplace = 1)
nbpredict <- predict(nbmodel, test_data[,1:12], type="class")

confusionMatrix(nbpredict,test_data$y)

library(caret)
library(klaR)
trainControl <- trainControl(method="cv", number=5)
fit.nb <- train(y~., data=train_data, method="nb",
                metric="Accuracy", trControl=trainControl)

predict.nb <- predict(fit.nb, test_data[,1:12])

confusionMatrix(predict.nb, test_data$y)
X <- varImp(fit.nb)
plot(X)

#praprocessing untuk ANN dan kNN
bank<- read.csv("Bank Marketing.csv")
#Merubah data menjadi numerrik
bank$housing <- as.character(bank$housing)
bank$housing[bank$housing == "yes"] <- "1"
bank$housing[bank$housing == "no"] <- "0"
bank$housing <- as.numeric(bank$housing)


bank$default <- ifelse(bank$default == "yes",1,0)
bank$loan <- ifelse(bank$loan == "yes",1,0)
bank$housing <- ifelse(bank$housing == "yes",1,0)
bank$marital <- ifelse(bank$marital == "married",1,0)
bank$job <- ifelse(bank$job == "unemployed"|bank$job == "retired",0,1)

bank$month <- factor(bank$month, 
                     levels = c("jan", "feb", "mar", "apr", 
                                "may", "jun", "jul", "aug", 
                                "sep", "oct", "nov", "dec"))
bank$month <- as.numeric(bank$month)

str(bank$contact)

bank$contact <- factor(bank$contact, 
                       levels = c("unknown","cellular","telephone"))
bank$contact <- as.numeric(bank$contact)

str(bank$education)
bank$education <- factor(bank$education, 
                         levels = c("unknown","primary","secondary", "tertiary"))
bank$education <- as.numeric(bank$contact)

str(bank$poutcome)
bank$poutcome <- ifelse(bank$poutcome == "succes",1,0)



#Membagi data train dan tes setelah dijadikan numerik
set.seed(1234)

sample_size <- floor(0.7*nrow(bank_drop))
train_index <- sample(seq_len(nrow(bank_drop)), size=sample_size)
train_data <- bank[train_index,]
test_data <- bank[-train_index,]


#ann yg lain harus numerik
library(nnet)
library(NeuralNetTools)
nn1 <- nnet(y~., data = train_data, size = 10, maxit = 300,
            trace = FALSE)

predict_nn1 <- predict(nn1, newdata = test_data[,1:16], type = 'class')
pred1 <- factor(predict_nn1)

confusionMatrix(pred1, test_data$y)

source('https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r')
pdf('./nn-example.pdf', width = 7, height = 7)
plot.nnet(nn1, alpha.val = 0.5, circle.col = list('lightgray', 'white'), bord.col = 'black')
dev.off()


#Normalizing data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
head(bank_normal)
bank_normal <- bank
bank_normal$ï..age <- normalize(bank_normal$ï..age)
bank_normal$balance <- normalize(bank_normal$balance)
bank_normal$duration <- normalize(bank_normal$duration)

#Membagi data train dan tes setelah dijadikan numerik dan di normalisasi
set.seed(1234)

sample_size <- floor(0.7*nrow(bank_drop))
train_index <- sample(seq_len(nrow(bank_drop)), size=sample_size)
train_data <- bank[train_index,]
test_data <- bank[-train_index,]

#mencari kNN yang optimal
NROW(train_data) #177 178

#kNN

knnModel212 <- knn3(y~., data = train_data, k = 177)
knnModel213 <- knn3(y~., data = train_data, k = 178)

knnPredict212 <- predict(knnModel212, test_data[,1:16], type="class")
knnPredict213 <- predict(knnModel213, test_data[,1:16], type="class")

table(test_data$y, knnPredict212)
table(test_data$y, knnPredict213)
tes <- confusionMatrix(knnPredict212, test_data$y)
confusionMatrix(knnPredict213, test_data$y)

#Mencari k yang paling tinggi dengan indikator akurasi
i = 173
k.opt = 173
for (i in 173:178) {
  knnModel <- knn3(y~., data = train_data, k = i)
  knnPredict <- predict(knnModel, test_data[,1:16], type="class")
  tes <- confusionMatrix(knnPredict, test_data$y)
  k.opt[i] <- tes$overall["Accuracy"]
  print(i)
  
}
k.opt
plot(k.opt, type = "b", xlim = c(174, 178), ylim = c(0.886, 0.888))



######################################## Repeated holdout tree 1 ########################################
# Repeated Holdout tree 1
library(rminer)
full_accuracy = 0
list_accuracy <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  dtreemodel_rpart <- rpart(y~., bank[H$tr,])
  predict_dtreerpart <- predict(dtreemodel_rpart, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(predict_dtreerpart, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  list_accuracy[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

########################################### Repeated Holdout tree 2#############################
library(rminer)
full_accuracy = 0
list_accuracy <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  dtreemodel_rpart <- rpart(y~., bank[H$tr,])
  ptree <- prune(dtreemodel_rpart,cp = dtreemodel_rpart$cptable[which.min(dtreemodel_rpart$cptable[,"xerror"]),"CP"])
  predict_dtreerpart <- predict(ptree, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(predict_dtreerpart, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  list_accuracy[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

rataan_akurasi_tree2 <- do.call(sum,list_accuracy)/NROW(list_accuracy)
rataan_akurasi_tree2

################################################## Cross Validation Tree 1 ####################################
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  tree <- rpart(y~., data = trainData)
  treepredict <- predict(tree, testData[,1:16], type="class")
  result <- confusionMatrix(treepredict, testData$y)
  accuracy <- result$overall['Accuracy']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")

################################################## END ##################################################

######################################## Repeated holdout buat Naive Bayes ########################################

# Repeated Holdout naive bayes
library(rminer)
full_accuracy = 0
list_accuracy <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  nbmodel <- naiveBayes(y~., data=bank_drop[H$tr,])
  nbpredict <- predict(nbmodel, bank_drop[H$ts,1:12], type="class")
  result<- confusionMatrix(nbpredict, bank_drop[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  list_accuracy[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

################################################## Cross Validation Naive bayes ####################################
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank_drop)), breaks=10, label=FALSE)
full_accuracy_cv = 0
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank_drop[testIndex, ]
  trainData <- bank_drop[-testIndex, ]
  nbmodel <- naiveBayes(y~., data=trainData)
  nbpredict <- predict(nbmodel, testData, type="class")
  result <- confusionMatrix(nbpredict, testData$y)
  accuracy <- result$overall['Accuracy']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")

######################################## Repeated holdout buat KNN ########################################

# Repeated Holdout naive bayes
library(rminer)
full_accuracy = 0
list_accuracy <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  knnmodel <- knn3(y~., data=bank_drop[H$tr,])
  knnpredict <- predict(knnmodel, bank_drop[H$ts,1:12], type="class")
  result<- confusionMatrix(knnpredict, bank_drop[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  list_accuracy[[i]] <- accuracy
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")
