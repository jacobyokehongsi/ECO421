# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 2021

@author: Jacob Yoke Hong Si
"""

# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


# Assign to "data"
df = pd.read_stata('banking.dta')
df['B20'].describe()
df['B3'].describe()
df['B4'].describe()
df['B5'].describe()
df['B26'].describe()
df['B27'].describe()
df['D1'].describe()
df['D2'].describe()
df['L8'].describe()

# Generate a dataframe
# The x-variables are storted in data.data
# The column names are stored in data.feature_names
# The y-variable is stored under data.target

# targets = df['B20']  # Add y-variable
# t_f = df[['B3', 'B4', 'B5', 'B26', 'B27', 'D1', 'D2', 'L8', 'B20']]
t_f = df[['B3', 'B4', 'B5', 'D1', 'D2', 'L8', 'B20']]
t_f = t_f.dropna()

features = t_f[:5]
targets = t_f['B20']

# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
targets_encoded = le.fit_transform(targets)

# features_lst = [df['B3'], df['B4'], df['B5'], df['B26'], df['B27'], df['D1'], df['D2'], df['L8']]
# features_encoded = []
# for f in features_lst:
#     features_encoded.append(le.fit_transform(f))

b3e = t_f['B3']
b4e = le.fit_transform(t_f['B4'])
b5e = le.fit_transform(t_f['B5'])
# b26e = le.fit_transform(t_f['B26'])
# b27e = le.fit_transform(t_f['B27'])
d1e = le.fit_transform(t_f['D1'])
d2e = le.fit_transform(t_f['D2'])
l8e = le.fit_transform(t_f['L8'])

# features_encoded = list(zip(b3e, b4e, b5e, b26e, b27e, d1e, d2e, l8e))
features_encoded = list(zip(b3e, b4e, b5e, d1e, d2e, l8e))

x_train, x_test, y_train, y_test = train_test_split(features_encoded, targets_encoded, test_size=0.3)

# ---------------------
# MODEL FITTING!!!!!
# ---------------------

# K-Nearest Neighbours 1,3,5

# Import the nearest neighbours classifier


# This loop estimates the k-nearest neighbors classifier
# for k=1, k=3, k=5
# It then calculates the score (percentage of correct predicitions)
# and the confusion matrix
# It saves the score, false positive rate, and recall rate

# score = []  # Create blank lists into which we'll save the score etc.
# false_pos = []
# recall = []

KNN = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(KNN, param_grid, cv=5)
#fit model to data
knn_gscv.fit(x_train, y_train)
print("Optimal K:", knn_gscv.best_params_)
#
# print('-------------------')
# print('For ', n, ' neighbours: ')
# print('')
s = knn_gscv.score(x_test, y_test)
print('The score is: ', s)  # Correct predictions in test data
print('')
cm = confusion_matrix(y_test, knn_gscv.predict(x_test))
print('The confusion matrix is: ')
print('')
print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                   columns=['Pred Neg', 'Pred Pos']))
print('')
print('--------------------')
# score.append(s)
# false_pos.append(cm[0, 1] / cm[0].sum())  # False positive rate
# recall.append(cm[1, 1] / cm[1].sum())  # Recall rate

# Logit Model



logit = LogisticRegression()
logit.fit(x_train, y_train)
print('-------------------')
print('Logistic Regression')
print('')
s = logit.score(x_test, y_test)
print('The score is: ', s)  # Correct predictions in test data
print('')
cm = confusion_matrix(y_test, logit.predict(x_test))
print('The confusion matrix is: ')
print('')
print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                   columns=['Pred Neg', 'Pred Pos']))
print('')
print('--------------------')
# score.append(s)
# false_pos.append(cm[0, 1] / cm[0].sum())
# recall.append(cm[1, 1] / cm[1].sum())

# Decision Tree



for d in [5, 10, 20, 40, 80]:
    print("Depth:", d)
    gini = DecisionTreeClassifier(criterion="gini", max_depth=d)
    gini = gini.fit(x_train, y_train)
    gini_predict = gini.predict(x_test)
    gini_accuracy = metrics.accuracy_score(y_test, gini_predict)
    print("Gini Accuracy:", gini_accuracy)
    # print("Gini Accuracy 2:", gini.score(x_test, y_test))
    cmg = confusion_matrix(y_test, gini_predict)
    print('The confusion matrix is: ')
    print('')
    print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                       columns=['Pred Neg', 'Pred Pos']))
    print('')
    print('--------------------')
    # score.append(s)
    # false_pos.append(cm[0, 1] / cm[0].sum())
    # recall.append(cm[1, 1] / cm[1].sum())

    entropy = DecisionTreeClassifier(criterion="entropy", max_depth=d)
    entropy.fit(x_train, y_train)
    entropy_predict = entropy.predict(x_test)
    entropy_accuracy = metrics.accuracy_score(y_test, entropy_predict)
    print("Information Gain Accuracy:", entropy_accuracy)
    # print("Information Gain Accuracy 2:", entropy.score(x_test, y_test))
    cme = confusion_matrix(y_test, entropy_predict)
    print('The confusion matrix is: ')
    print('')
    print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                       columns=['Pred Neg', 'Pred Pos']))
    print('')
    print('--------------------')

# Random Forest

rf = RandomForestClassifier()
# Fit on training data
rf.fit(x_train, y_train)
print('-------------------')
print('Random Forest')
print('')
s = rf.score(x_test, y_test)
print('The score is: ', s)  # Correct predictions in test data
print('')
cm = confusion_matrix(y_test, rf.predict(x_test))
print('The confusion matrix is: ')
print('')
print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                   columns=['Pred Neg', 'Pred Pos']))
print('')
print('--------------------')
