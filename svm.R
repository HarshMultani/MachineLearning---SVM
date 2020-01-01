# SVM Algorithm
install.packages('caTools')
library(caTools)
install.packages('e1071')
library(e1071)

# Import the dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[3:5]


# Splitting the dataset into train and test set
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Standard Scaling our dataset
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


# Fitting SVM model
model <- svm(Purchased ~ . , data = training_set, type = 'C-classification', kernel = 'linear')


# Predicting the test set resulta
Y_Pred <- predict(model, newdata = test_set[-3])


# Making the confusion matrix
cm = table(test_set[,3], Y_Pred)
