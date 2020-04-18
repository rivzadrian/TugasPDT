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

col_to_drop <- c("job","marital","education","default","marital","loan","contact","month","poutcome","y","housing","day")
bank_hist <- bank[,!(names(bank) %in% col_to_drop)]

#bank$y <- as.character(bank$y)
#bank$y[bank$y == "yes"] <- "1"
#bank$y[bank$y == "no"] <- "0"
#bank$y <- as.numeric(bank$y)






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
bank <- read.csv("Bank Marketing.csv")
#Membagi data train dan test
set.seed(1234)

sample_size <- floor(0.7*nrow(bank))
train_index <- sample(seq_len(nrow(bank)), size=sample_size)
train_data <- bank[train_index,]
test_data <- bank[-train_index,]
x = train_data[,-17]
y = train_data$y

#ANN harus numerik
data_test_2 <- test_data %>% mutate(y = ifelse(y == "no", 0, 1))

library(neuralnet)
NN <- neuralnet(y~., data = train_data, linear.output = FALSE,
                hidden = 5, rep = 10, threshold = 0.1)
plot(NN, rep = "best")

predict_nn <- neuralnet::compute(NN, test_data[,1:17])
nn_result1 <- data.frame(actual = data_test_2$y,
                         prediction = predict_nn$net.result)

rounded_result <- sapply(nn_result1, round, digits = 0)
rounded_resultdf <- data.frame(rounded_result)
attach(rounded_resultdf)

table(actual, prediction.2)

actualfactor <- as.factor(actual)
predfactor <- as.factor(prediction.2)

confusionMatrix(actualfactor, predfactor)

#pake ann yg lain harus numerik
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

#Decision tree
library(rpart)
library(rpart.plot)
library(caret)
dtreemodel_rpart <- rpart(y~., train_data)
predict_dtreerpart <- predict(dtreemodel_rpart, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)
printcp(dtreemodel_rpart)
plotcp(dtreemodel_rpart)
rpart.plot(dtreemodel_rpart)

ptree <- prune(dtreemodel_rpart,cp = dtreemodel_rpart$cptable[which.min(dtreemodel_rpart$cptable[,"xerror"]),"CP"])
rpart.plot(ptree, uniform=TRUE, main="Pruned Classification Tree")
printcp(ptree)
plotcp(ptree)
predict_dtreerpart <- predict(ptree, test_data[,1:16], type="class")
confusionMatrix(predict_dtreerpart, test_data$y)
#Naive bayes
#Building a model
#split data into training and test data sets
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

col_to_drop <- c("ï..age","age","month","day","marital","job")
bank_drop <- bank[,!(names(bank) %in% col_to_drop)]

#Misah data train dan test
set.seed(1234)

sample_size <- floor(0.7*nrow(bank_drop))
train_index <- sample(seq_len(nrow(bank_drop)), size=sample_size)
train_data <- bank_drop[train_index,]
test_data <- bank_drop[-train_index,]
x = train_data[,-11]
y = train_data$y

missmap(bank)
#naive bayes yg praktikum
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


data2$Age <- as.numeric(data2$Age)



#praprocessing untuk ANN dan kNN

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
train_data <- bank_drop[train_index,]
test_data <- bank_drop[-train_index,]
x = train_data[,-11]
y = train_data$y



#ANN
data_test_2 <- test_data %>% mutate(y = ifelse(y == "no", 0, 1))

library(neuralnet)
NN <- neuralnet(y~., data = train_data, linear.output = FALSE,
                hidden = 5, rep = 10, threshold = 0.3)
plot(NN, rep = "best")

predict_nn <- neuralnet::compute(NN, test_data[,1:17])
nn_result1 <- data.frame(actual = data_test_2$y,
                         prediction = predict_nn$net.result)

rounded_result <- sapply(nn_result1, round, digits = 0)
rounded_resultdf <- data.frame(rounded_result)
attach(rounded_resultdf)

table(actual, prediction.2)

actualfactor <- as.factor(actual)
predfactor <- as.factor(prediction.2)

confusionMatrix(actualfactor, predfactor)
