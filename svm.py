# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 00:09:29 2020

@author: 138709
"""

# SVM Algorithm

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values


# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Feature Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to our dataset
from sklearn.svm import SVC
model = SVC(kernel = 'linear', random_state = 0)
model.fit(X_train, Y_train)


# Predicting the test set results
Y_pred = model.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
