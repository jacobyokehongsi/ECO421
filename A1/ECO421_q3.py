# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 2021

@author: Jacob Yoke Hong Si
"""

# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti

# Data Cleaning
# Original Dataset with Selected Variables
df = pd.read_stata('banking.dta')

t_f = df[['B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4',
          'n_B7_5', 'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7',
          'B20']]

print(t_f['B20'].describe())
print(t_f['B20'].isna().sum())

for (colname, colval) in t_f.iteritems():
    print(t_f[colname].describe())

t_f = t_f.dropna()
le = preprocessing.LabelEncoder()
t_f = t_f.apply(le.fit_transform)
features = t_f[['B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4', 'n_B7_5', 'n_B7_6',
                'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7']]
targets = t_f['B20']


# Original Dataset with Selected Variables + Married Individuals
t_f_married = df[['B20', 'B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4', 'n_B7_5',
                  'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7',
                  'n_C1_2']]
t_f_married = t_f_married.dropna()
t_f_married = t_f_married.apply(le.fit_transform)
t_f_married = t_f_married[t_f_married['B20'] == 0]  # married without a bank account
features_married = t_f_married[['B20', 'B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4',
                                'n_B7_5', 'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5',
                                'n_B25_6', 'n_B25_7']]
targets_married = t_f_married['n_C1_2']


# # ---------------------
# # MODEL FITTING!!!!!
# # ---------------------

def best_knn(x, y):
    KNN = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(KNN, param_grid, cv=5)
    knn_gscv.fit(x, y)
    print("Optimal K:", knn_gscv.best_params_)
    return knn_gscv.best_params_["n_neighbors"]


def tree_interpreter(tree_model, x_married, y_married):
    x_train, x_test, y_train, y_test = train_test_split(x_married, y_married, test_size=0.2)
    tree_model.fit(x_train.values, y_train)
    instance = x_test[x_test['B20'] == 0]
    prediction, bias, contributions = ti.predict(tree_model, instance.values)
    print("Feature contributions:")
    for c, feature in zip(contributions[0],
                          list(x_married.columns)):
        print(feature, c)


def ex2_1(models, x, y, x_married, y_married):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    auc_scores = []
    for m in models:
        print("Results for", m)
        if m == KNN:
            k = best_knn(x.values, y)
            m = KNeighborsClassifier(n_neighbors=k)
        m.fit(x_train.values, y_train)
        fpr, tpr, thresh = roc_curve(y_test, m.predict_proba(x_test.values)[:, 1])
        plt.plot(fpr, tpr, fpr, fpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()

        auc = roc_auc_score(y_test, m.predict_proba(x_test.values)[:, 1])
        auc_scores.append(auc)
        print('Area under curve is: ', round(auc, 3))

        cm = confusion_matrix(y_test, m.predict(x_test.values))
        print('The confusion matrix is: ')
        print('')
        print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                           columns=['Pred Neg', 'Pred Pos']))
        print('')

        if m == tree:
            tree_interpreter(m, x_married, y_married)

        if m == logit:
            importance = m.coef_[0]
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: {}, Score: {}'.format(features.columns[i], v))
            # plot feature importance
            plt.bar([x for x in range(len(importance))], importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance")
            plt.show()

        print('===================================================')

    print("AUC scores for knn, logit, tree and rf respectively are:", auc_scores)


if __name__ == "__main__":
    np.random.seed(0)
    KNN = KNeighborsClassifier()
    logit = LogisticRegression(solver='newton-cg')
    tree = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    models = [KNN, logit, tree, rf]
    ex2_1(models, features, targets, features_married, targets_married)


