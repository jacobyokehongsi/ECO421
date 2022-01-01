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
df = pd.read_stata('banking.dta')
# df['B20'].describe()
# df['B3'].describe()
# df['B4'].describe()
# df['B5'].describe()
# # df['B26'].describe()
# # df['B27'].describe()
# df['D1'].describe()
# df['D2'].describe()
# df['L8'].describe()

t_f = df[['B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4',
          'n_B7_5', 'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7',
          'B20']]

# print(t_f.isna().sum())
t_f = t_f.dropna()
le = preprocessing.LabelEncoder()
t_f = t_f.apply(le.fit_transform)
features = t_f[['B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4', 'n_B7_5', 'n_B7_6',
                'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7']]
targets = t_f['B20']

######################################### Married ###############################
# print(len(df['n_C1_2']))
# print(df['n_C1_2'].isna().sum())

t_f_married = df[['n_C1_2', 'B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4', 'n_B7_5',
                  'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5', 'n_B25_6', 'n_B25_7',
                  'B20']]

# print(t_f.isna().sum())
t_f_married = t_f_married.dropna()
t_f_married = t_f_married.apply(le.fit_transform)
features_married = t_f_married[['n_C1_2', 'B2', 'n_B3_age', 'B4', 'B5', 'B10', 'n_B7_1', 'n_B7_2', 'n_B7_3', 'n_B7_4',
                                'n_B7_5', 'n_B7_6', 'n_B7_7', 'n_B25_1', 'n_B25_2', 'n_B25_3', 'n_B25_4', 'n_B25_5',
                                'n_B25_6', 'n_B25_7']]
targets_married = t_f_married['B20']


############################################# Data Cleaning ###########################################################


# x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)
#
# # ---------------------
# # MODEL FITTING!!!!!
# # ---------------------
#
# KNN = KNeighborsClassifier()
# # create a dictionary of all values we want to test for n_neighbors
# param_grid = {'n_neighbors': np.arange(1, 25)}
# # use gridsearch to test all values for n_neighbors
# knn_gscv = GridSearchCV(KNN, param_grid, cv=5)
# # fit model to data
# knn_gscv.fit(x_train.values, y_train)
# print("Optimal K:", knn_gscv.best_params_)
#
# KNN = KNeighborsClassifier(n_neighbors=1)
# KNN.fit(x_train.values, y_train)
# knn_predict = KNN.predict_proba(x_test.values)
#
# s = KNN.score(x_test.values, y_test)
# print('The score is: ', s)  # Correct predictions in test data
# print('')
# cm = confusion_matrix(y_test, KNN.predict(x_test.values))
# print('The confusion matrix is: ')
# print('')
# print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
#                    columns=['Pred Neg', 'Pred Pos']))
# print('')
# print('--------------------')
#
# # Logit Model
# logit = LogisticRegression(solver='newton-cg')
# logit.fit(x_train.values, y_train)
# logit_predict = logit.predict_proba(x_test.values)
#
# # Decision Tree
# for d in [5, 10, 20, 40, 80]:
#     print("Depth:", d)
#     gini = DecisionTreeClassifier(criterion="gini", max_depth=d)
#     gini = gini.fit(x_train.values, y_train)
#     gini_predict = gini.predict(x_test.values)
#     gini_accuracy = metrics.accuracy_score(y_test, gini_predict)
#     print("Gini Accuracy:", gini_accuracy)
#     # print("Gini Accuracy 2:", gini.score(x_test, y_test))
#     # score.append(s)
#     # false_pos.append(cm[0, 1] / cm[0].sum())
#     # recall.append(cm[1, 1] / cm[1].sum())
#
#     entropy = DecisionTreeClassifier(criterion="entropy", max_depth=d)
#     entropy.fit(x_train.values, y_train)
#     entropy_predict = entropy.predict(x_test.values)
#     entropy_accuracy = metrics.accuracy_score(y_test, entropy_predict)
#     print("Information Gain Accuracy:", entropy_accuracy)
#     # print("Information Gain Accuracy 2:", entropy.score(x_test, y_test))
#     print('')
#     print('--------------------')
#
# tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)  # highest accuracy
# tree = tree.fit(x_train.values, y_train)
# tree_predict = tree.predict_proba(x_test.values)
#
# prediction, bias, contributions = ti.predict(tree, x_test.values)
#
# # Random Forest
#
# rf = RandomForestClassifier()
# rf.fit(x_train.values, y_train)
# rf_predict = rf.predict_proba(x_test.values)

# auc_knn = roc_auc_score(y_test, knn_predict[:, 1])
# auc_logit = roc_auc_score(y_test, logit_predict[:, 1])
# auc_tree = roc_auc_score(y_test, tree_predict[:, 1])
# auc_rf = roc_auc_score(y_test, rf_predict[:, 1])

def best_knn(x, y):
    KNN = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(KNN, param_grid, cv=5)
    knn_gscv.fit(x, y)
    print("Optimal K:", knn_gscv.best_params_)
    return knn_gscv.best_params_["n_neighbors"]
#
#
# print("Married prediction:", tree_model.predict(married))
# prediction, bias, contributions = ti.predict(m, married)
# for i in range(len(married)):
#     print("Instance", i)
#     print("Bias (trainset mean)", bias[i])
#     print("Feature contributions:")
#     for c, feature in sorted(zip(contributions[i], list(x.columns)), key=lambda x: -abs(x[0])):
#         print(feature, round(c, 2))
#
# def tree_interpreter(tree_model, x, y):
#     return


def ex2_1(models, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    for m in models:
        print("Results for", m)
        if m == KNN:
            k = best_knn(x.values, y)
            m = KNeighborsClassifier(n_neighbors=k)
        m.fit(x_train.values, y_train)
        # cv = cross_val_score(model, x, y, cv=5)
        fpr, tpr, thresh = roc_curve(y_test, m.predict_proba(x_test.values)[:, 1])
        plt.plot(fpr, tpr, fpr, fpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()

        auc = roc_auc_score(y_test, m.predict_proba(x_test.values)[:, 1])
        print('Area under curve is: ', round(auc, 3))

        cm = confusion_matrix(y_test, m.predict(x_test.values))
        print('The confusion matrix is: ')
        print('')
        print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'],
                           columns=['Pred Neg', 'Pred Pos']))
        print('')
        print('--------------------')
        # print('Cross validation score is :', round(cv.mean(), 2))

        if m == tree:
            x_train, x_test, y_train, y_test = train_test_split(features_married, targets_married, test_size=0.2)
            tree.fit(x_train.values, y_train)
            # tree.fit(features_married.values, targets_married)
            instance = x_test[x_test['n_C1_2'] == 1]
            # instance = features_married[features_married['n_C1_2'] == 1]
            # print(tree.predict_proba(instance.values))
            prediction, bias, contributions = ti.predict(tree, instance.values)
            # print("Prediction", prediction)
            # print("Bias (trainset prior)", bias)
            print("Feature contributions:")
            for c, feature in zip(contributions[0],
                                  list(features_married.columns)):
                print(feature, c)


if __name__ == "__main__":
    np.random.seed(0)
    KNN = KNeighborsClassifier()
    logit = LogisticRegression(solver='newton-cg')
    # tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tree = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    models = [KNN, logit, tree, rf]
    # models = [tree]

    # scores = []
    # for i in np.arange(1, 25):
    #     KNN = KNeighborsClassifier(n_neighbors=i)
    #     CV = cross_val_score(KNN, features.values, targets, cv=5)
    #     CV = CV.mean()
    #     scores.append(CV)
    # plt.plot(np.arange(1, 25), scores)
    # plt.show()

    ex2_1(models, features, targets)




    # print("KNN Model")
    # ex2_1(KNN, features, targets)
    # print("Logit Model")
    # ex2_1(logit, features, targets)
    # print("Decision Tree Model")
    # ex2_1(tree, features, targets)
    # print("Random Forest Model")
    # ex2_1(rf, features, targets)





# print(auc_knn, auc_logit, auc_tree, auc_rf)

# RocCurveDisplay.from_estimator(KNN, x_test.values, y_test)
# RocCurveDisplay.from_estimator(logit, x_test.values, y_test)
# RocCurveDisplay.from_estimator(tree, x_test.values, y_test)
# RocCurveDisplay.from_estimator(rf, x_test.values, y_test)
# plt.show()
