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
library(nnet)
library(e1071)

####################################### EKSPLORASI DATA ####################################################
#Mengubah data frame
bank <- read.csv("Bank Marketing.csv")
bank <- bank %>% 
  rename(
    age = ï..age
  )
bank <- data.frame(bank)
View (bank)
head(bank)
#Deklarasi Atribut 
bank$age <- as.numeric(bank$age)
bank$balance <- as.numeric(bank$balance)
bank$day <- as.numeric(bank$day)
bank$duration <- as.numeric(bank$duration)
bank$campaign <- as.numeric(bank$campaign)
bank$pdays <- as.numeric(bank$pdays)
bank$poutcome <- as.numeric(bank$poutcome)

bank$job <- as.factor(bank$job)
bank$month <- as.ordered(bank$month)
bank$education <- as.factor(bank$education)
bank$loan <- as.factor(bank$loan)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$y <- as.factor(bank$y)

#Melihat struktur data 
str(bank)

#Mencari head and Tail
head(bank)
tail(bank)

#Mengetahui Summary data
summary(bank)

#Mencari duplikasi data
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

#Perbandingan jumlah kelas secara keseluruhan data 
##Visualisasi Jumlah Class
plot(x=bank$y, ylim=c(0,50000), xlab="Class Y", ylab="count")

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

# heatmap
corr <- round(cor(bank[,c(1,6,10,12,13,14,15)]), 2)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Bank Marketing", 
           ggtheme=theme_bw)

# Plot probabilitas berlangganan
plot(x=bank$contact, y=bank$y, xlab = "contact", ylab="Probabilitas berlangganan", ylim=c(0,1))
plot(x=bank$marital, y=bank$y, xlab = "marital", ylab="Probabilitas berlangganan", ylim=c(0,1))
plot(x=bank$job, y=bank$y, xlab = "job", ylab="Probabilitas berlangganan", ylim=c(0,1))

########################################## Membagi data train dan test################################
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

########################################## Decision tree Gini###############################

dtreemodel_rpart_gini <- rpart(y~., train_data, parms = list(split = "gini"))
predict_dtreerpart_gini <- predict(dtreemodel_rpart_gini, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart_gini, test_data$y)
rpart.plot(dtreemodel_rpart_gini, main = "gini Index")#Menampilkan tree
printcp(dtreemodel_rpart_gini) # mengetahui CP dari tree
plotcp(dtreemodel_rpart_gini)# Menampilkan plot XP dan X-Val realtive error
resultdt <- table(predict_dtreerpart_gini, test_data$y)
cmdt <- confusionMatrix(predict_dtreerpart_gini, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

########################################## Prune tree dengan indikator CP Gini#######################
ptree <- prune(dtreemodel_rpart_gini,cp = dtreemodel_rpart_gini$cptable[which.min(dtreemodel_rpart_gini$cptable[,"xerror"]),"CP"])
rpart.plot(ptree, uniform=TRUE, main="Pruned Classification Gini Tree")
printcp(ptree)
plotcp(ptree)
predict_dtreerpart <- predict(ptree, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)


########################################## Decision tree entrophy##########################
dtreemodel_rpart_entro <- rpart(y~., train_data, parms = list(split = "information"))
predict_dtreerpart_entro <- predict(dtreemodel_rpart_entro, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart_entro, test_data$y)
printcp(dtreemodel_rpart_entro)# mengetahui CP dari tree
plotcp(dtreemodel_rpart_entro)# Menampilkan plot XP dan X-Val realtive error
rpart.plot(dtreemodel_rpart_entro, main = "Entrophy")#Menampilkan tree
resultdt <- table(predict_dtreerpart_entro, test_data$y)
cmdt <- confusionMatrix(predict_dtreerpart_entro, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

########################################## Prune tree dengan indikator CP Entrophy###################
ptree <- prune(dtreemodel_rpart_entro,cp = dtreemodel_rpart_entro$cptable[which.min(dtreemodel_rpart_entro$cptable[,"xerror"]),"CP"])
rpart.plot(ptree, uniform=TRUE, main="Pruned Classification Entrophy Tree")
printcp(ptree)
plotcp(ptree)
predict_dtreerpart <- predict(ptree, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)


########################################## Naive bayes tes siginifikansi atribut############################
#Building a model

library(e1071)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
Predict <- predict(model,newdata = test_data )
X <- varImp(model)
plot(X)

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

head(bank)
#Drop kolom yang tidak signifikan dan membagi kembali data tes dan training
col_to_drop <- c("age","age","month","day","marital","job")
bank_drop <- bank[,!(names(bank) %in% col_to_drop)]

#Misah data train dan test
set.seed(1234)

sample_size <- floor(0.7*nrow(bank_drop))
train_index <- sample(seq_len(nrow(bank_drop)), size=sample_size)
train_data <- bank_drop[train_index,]
test_data <- bank_drop[-train_index,]
x = train_data[,-11]
y = train_data$y


########################################## naive bayes no laplace###########################
nbmodel <- naiveBayes(y~., data=train_data, laplace = 0)
nbpredict <- predict(nbmodel, test_data[,1:12], type="class")

confusionMatrix(nbpredict,test_data$y)

resultdt <- table(nbpredict, test_data$y)
cmdt <- confusionMatrix(nbpredict, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

########################################## naive bayes laplace##############################
nbmodel <- naiveBayes(y~., data=train_data, laplace = 1)
nbpredict <- predict(nbmodel, test_data[,1:12], type="class")

confusionMatrix(nbpredict,test_data$y)

resultdt <- table(nbpredict, test_data$y)
cmdt <- confusionMatrix(nbpredict, test_data$y)
evaluation(resultdt, accuracy = cmdt)#Menampilakn precision recall dan f-measure

########################################## praprocessing untuk ANN dan kNN#################################
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

bank$contact <- factor(bank$contact, 
                       levels = c("unknown","cellular","telephone"))
bank$contact <- as.numeric(bank$contact)

bank$education <- factor(bank$education, 
                         levels = c("unknown","primary","secondary", "tertiary"))
bank$education <- as.numeric(bank$contact)


bank$poutcome <- ifelse(bank$poutcome == "succes",1,0)



#Membagi data train dan tes setelah dijadikan numerik
set.seed(1234)

sample_size <- floor(0.7*nrow(bank))
train_index <- sample(seq_len(nrow(bank)), size=sample_size)
train_data <- bank[train_index,]
test_data <- bank[-train_index,]


########################################## ANN###################################################
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


########################################## Normalizing data untuk kNN##########################
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
head(bank_normal)
bank_normal <- bank
bank_normal$age <- normalize(bank_normal$age)
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

########################################## kNN########################################

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



############################################### END Masuk Evaluasi ##################################################
########################################## Repeated holdout tree Gini ########################################
# Repeated Holdout tree 1
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  dtreemodel_rpart <- rpart(y~., bank[H$tr,], parms = list(split = "gini"))
  predict_dtreerpart <- predict(dtreemodel_rpart, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(predict_dtreerpart, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

########################################## Repeated holdout tree Entrophy ########################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  dtreemodel_rpart <- rpart(y~., bank[H$tr,], parms = list(split = "information"))
  predict_dtreerpart <- predict(dtreemodel_rpart, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(predict_dtreerpart, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")
########################################## Repeated Holdout tree Gini prunned#############################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  dtreemodel_rpart <- rpart(y~., bank[H$tr,], parms = list(split = "information"))
  ptree <- prune(dtreemodel_rpart,cp = dtreemodel_rpart$cptable[which.min(dtreemodel_rpart$cptable[,"xerror"]),"CP"])
  predict_dtreerpart <- predict(ptree, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(predict_dtreerpart, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

rataan_akurasi_tree2 <- do.call(sum,list_accuracy)/NROW(list_accuracy)
rataan_akurasi_tree2

########################################## Cross Validation Tree Gini ####################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  tree <- rpart(y~., data = trainData)
  treepredict <- predict(tree, testData[,1:16], type="class")
  result <- confusionMatrix(treepredict, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")

########################################## Cross Validation Tree Entrophy ####################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  tree <- rpart(y~., data = trainData, parms = list(split = "information") )
  treepredict <- predict(tree, testData[,1:16], type="class")
  result <- confusionMatrix(treepredict, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")
########################################## Cross Validation Tree Gini prunned ####################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini (baris 16-27)
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  tree <- rpart(y~., data = trainData)
  ptree <- prune(tree,cp = tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
  treepredict <- predict(ptree, testData[,1:16], type="class")
  result <- confusionMatrix(treepredict, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")


########################################## Repeated holdout buat Naive Bayes No Laplace ########################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini
# Repeated Holdout naive bayes
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  nbmodel <- naiveBayes(y~., data=bank_drop[H$tr,])
  nbpredict <- predict(nbmodel, bank_drop[H$ts,1:12], type="class")
  result<- confusionMatrix(nbpredict, bank_drop[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

########################################## Repeated holdout Naive Bayes Laplace ########################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini
# Repeated Holdout naive bayes
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  nbmodel <- naiveBayes(y~., data=bank_drop[H$tr,], laplace = 1)
  nbpredict <- predict(nbmodel, bank_drop[H$ts,1:12], type="class")
  result<- confusionMatrix(nbpredict, bank_drop[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")

########################################## Cross Validation Naive bayes No Laplace ####################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank_drop)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank_drop[testIndex, ]
  trainData <- bank_drop[-testIndex, ]
  nbmodel <- naiveBayes(y~., data=trainData)
  nbpredict <- predict(nbmodel, testData, type="class")
  result <- confusionMatrix(nbpredict, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")

########################################## Cross Validation Naive bayes Laplace ####################################
#ingat ubah data bank menjadi normal kembali untuk menjalankan ini
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank_drop)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank_drop[testIndex, ]
  trainData <- bank_drop[-testIndex, ]
  nbmodel <- naiveBayes(y~., data=trainData, laplace = 1)
  nbpredict <- predict(nbmodel, testData, type="class")
  result <- confusionMatrix(nbpredict, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")

########################################## Repeated holdout buat kNN k = 177########################################
#ingat ubah data bank menjadi NUMERIK kembali untuk menjalankan ini
# Repeated Holdout KNN
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()

for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  knnmodel <- knn3(y~., data=bank[H$tr,], k = 177)
  knnpredict <- predict(knnmodel, bank[H$ts,1:16], type="class")
  result<- confusionMatrix(knnpredict, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}

k =### Plot
cat("Tree: ",
      "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
      "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")
       
########################################## Cross Validation kNN k = 177 ####################################
#ingat ubah data bank menjadi NUMERIK kembali untuk menjalankan ini
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  knnModel212 <- knn3(y~., data = trainData, k = 177)
  knnPredict212 <- predict(knnModel212, testData[,1:16], type="class")
  result <- confusionMatrix(knnPredict212, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")
########################################## Repeated holdout buat ANN ########################################
#ingat ubah data bank menjadi NUMERIK kembali untuk menjalankan ini
# Repeated Holdout ANN
library(rminer)
full_accuracy = 0
full_precision = 0
full_recall = 0
full_fval = 0
list_accuracy <- list()
list_precision <- list()
list_recall <- list()
list_fval <- list()
for(i in 1:100) {
  H = holdout(bank$y, ratio = 2/3, mode="random", seed = NULL)
  nn1 <- nnet(y~., data = bank[H$tr,], size = 10, maxit = 300,
              trace = FALSE)
  predict_nn1 <- predict(nn1, newdata = bank[H$ts,1:16], type = 'class')
  pred1 <- factor(predict_nn1)
  result<- confusionMatrix(pred1, bank[H$ts,]$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy = full_accuracy + accuracy
  full_precision = full_precision + precision
  full_recall = full_recall + recall
  full_fval = full_fval + fval
  list_accuracy[[i]] <- accuracy
  list_precision[[i]] <- precision
  list_recall[[i]] <- recall
  list_fval[[i]] <- fval
}


### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy/100, "Precision: ", full_precision/100, 
    "Recall: ", full_recall/100,"F-Measure: ", full_fval/100,"\n")
x <- c(1:100)
plot(x, list_accuracy, type="o")


########################################## Cross Validation ANN ####################################
#ingat ubah data bank menjadi NUMERIK kembali untuk menjalankan ini
### Membuat 10 fold
folds <- cut(seq(1,nrow(bank)), breaks=10, label=FALSE)
full_accuracy_cv = 0
full_precision_cv = 0
full_recall_cv = 0
full_fval_cv = 0
list_precision_cv <- list()
list_recall_cv <- list()
list_fval_cv <- list()
list_accuracy_cv <- list()

for(i in 1:10) {
  testIndex <- which(folds == i, arr.ind = TRUE)
  testData <- bank[testIndex, ]
  trainData <- bank[-testIndex, ]
  nn1 <- nnet(y~., data = trainData, size = 10, maxit = 300,
              trace = FALSE)
  predict_nn1 <- predict(nn1, newdata = testData[,1:16], type = 'class')
  pred1 <- factor(predict_nn1)
  result <- confusionMatrix(pred1, testData$y)
  accuracy <- result$overall['Accuracy']
  precision <- result$byClass['Precision']
  recall <- result$byClass['Recall']
  fval <- result$byClass['F1']
  
  cat("batch: ",i,
      "accuracy: ",accuracy,"\n")
  full_accuracy_cv = full_accuracy_cv + accuracy
  list_accuracy_cv[[i]] <- accuracy
  full_precision_cv = full_precision_cv + precision
  full_recall_cv = full_recall_cv + recall
  full_fval_cv = full_fval_cv + fval
  list_accuracy_cv[[i]] <- accuracy
  list_precision_cv[[i]] <- precision
  list_recall_cv[[i]] <- recall
  list_fval_cv[[i]] <- fval
}

### Plot
cat("Tree: ",
    "Accuracy: ", full_accuracy_cv/10, "Precision: ", full_precision_cv/10, 
    "Recall: ", full_recall_cv/10,"F-Measure: ", full_fval_cv/10,"\n")
x <- c(1:10)
plot(x, list_accuracy_cv, type="o")
