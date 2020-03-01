#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:08:58 2019

@author: dharmit
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from yellowbrick.classifier import ClassificationReport
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# to avoid the warning messages
pd.options.mode.chained_assignment = None

heart_data = pd.read_csv("heart.csv")

# select columns other than 'target'
cols = [col for col in heart_data.columns if col not in ['target']]

# dropping the 'target column
data = heart_data[cols]

#assigning the target column as target
target = heart_data['target']

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)


#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())

#print the accuracy score of the model
print("Accuracy using Naive-Bayes : ",round(accuracy_score(target_test, pred, normalize = True)*100,2),"%")



# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['1','0'])

visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.poof() 


#import the necessary modules


#create an object of type LinearSVC
svc_model = SVC(kernel='linear')

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)

#print the accuracy score of the model
print("Accuracy using LinearSVC : ",round(accuracy_score(target_test, pred, normalize = True)*100,2),"%")


# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svc_model, classes=['1','0'])

visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(data_train, target_train)

#Predict the response for test dataset
y_pred = model.predict(data_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using Adaboost:",round(metrics.accuracy_score(target_test, y_pred)*100,2),"%")

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(model, classes=['1','0'])

visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# Create classifer object
rfc = RandomForestClassifier()
# Train RandomForest Classifer
mod = rfc.fit(data_train, target_train)

#Predict the response for test dataset
y_pred = mod.predict(data_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using random forest:",round(metrics.accuracy_score(target_test, y_pred)*100,2),"%")


# Instantiate the classification model and visualizer
visualizer = ClassificationReport(mod, classes=['1','0'])

visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# Create classifer object
lrc = LogisticRegression()
# Train LogisticRegression Classifer
mod1 = lrc.fit(data_train, target_train)

#Predict the response for test dataset
y_pred = mod1.predict(data_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using Logistic Regression:",round(metrics.accuracy_score(target_test, y_pred)*100,2),"%")

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(mod1, classes=['1','0'])

visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data