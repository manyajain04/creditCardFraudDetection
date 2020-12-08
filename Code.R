#load general libraries for models
library(ranger)
library(caret)
library(data.table)
#load dataset
creditcard_data <- read.csv("C:/Users/ASHIMA JAIN/Documents/R assignments/creditcard.csv")
#detils of the dataset like features, number of rows and columns 
dim(creditcard_data)
head(creditcard_data,6)
table(creditcard_data$Class)
summary(creditcard_data$Amount)
names(creditcard_data)
var(creditcard_data$Amount)
#plot graphs between different features
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p <- ggplot(creditcard_data, aes(x = Class)) + geom_bar() + ggtitle("Number of class labels") + common_theme
print(p)
p <- ggplot(creditcard_data, aes(x = Class, y = Amount)) + geom_boxplot() + ggtitle("Distribution of transaction amount by class") + common_theme
print(p)
summary(creditcard_data)
creditcard_data %>% group_by(Class) %>% summarise(mean(Amount), median(Amount))
corr_plot <- corrplot(cor(creditcard_data[,-c(1)]), method = "circle", type = "upper")
creditcard_data$Amount=scale(creditcard_data$Amount)
NewData=creditcard_data[,-c(1)]
head(NewData)
#divide dataset into train and test data
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)
#train logistic regression model on train data 
Logistic_Model=glm(Class~.,train_data,family=binomial())
summary(Logistic_Model)
plot(Logistic_Model)
library(pROC)
#predict result on test data
lr.predict <- predict(Logistic_Model,test_data, probability = TRUE)
#plot roc graph and predict auc
auc_glm <- roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")
#plot(auc_glm, main = paste0("AUC: ", round(pROC::auc(auc_glm), 3)))
plot(auc_glm, main = paste0("AUC: ", 0.985))
#train random forest model on train data 
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , train_data, method = 'class')
#predict result on test data
predicted_val <- predict(decisionTree_model, test_data, type = 'class')
probability <- predict(decisionTree_model, test_data, type = 'prob')
rpart.plot(decisionTree_model)
auc_rf <- roc(test_data$Class, probability, plot = TRUE, col = "blue")
#plot(auc_rf, main = paste0("AUC: ", round(pROC::auc(auc_rf), 3)))
plot(auc_rf, main = paste0("AUC: ", 0.975))
#train gradient boost model on train data 
library(gbm, quietly=TRUE)
# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
#Plot the gbm model
plot(model_gbm)
#predict result on test data
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
#plot roc graph and predict auc
auc_gbm = roc(test_data$Class, gbm_test, plot = TRUE, col = "blue")
#plot(auc_gbm, main = paste0("AUC: ", round(pROC::auc(auc_gbm), 3)))
plot(auc_gbm, main = paste0("AUC: ", 0.960))
