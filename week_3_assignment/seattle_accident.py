# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:33:49 2020

@author: MY HP
"""

# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing the data

data = pd.read_csv("G:/DataSets/Seattle Accident Data/Data-Collisions.csv")
print("data is imported")

feature = data[["SEVERITYCODE","WEATHER","ROADCOND", "LIGHTCOND"]]

cleaned_data = feature.dropna()

x = cleaned_data.iloc[: ,1:4]
y = cleaned_data.iloc[:, 0]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(drop = "first")
x["WEATHER"] = labelencoder.fit_transform(x["WEATHER"])
x["ROADCOND"] = labelencoder.fit_transform(x["ROADCOND"])
x["LIGHTCOND"] = labelencoder.fit_transform(x["LIGHTCOND"])

x = onehotencoder.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

# logistic Regresson

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)



# FOR TESTING
min(189337)
#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = "mle", svd_solver = "full")
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


#  Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(x_train,y_train)

#predecting the result

y_pred_lr = lr.predict(x_test)

# checking the accuracy of model
# confusion matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_lr)

# jaccard score
from sklearn.metrics import jaccard_score
js_lr = jaccard_score(y_test, y_pred_lr)

# f1_score
from sklearn.metrics import f1_score
f1_lr = f1_score(y_test, y_pred_lr)

# log_loss
from sklearn.metrics import log_loss
l_los_lr = log_loss(y_test, y_pred_lr)

# K_nearest_neighbour

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

# predicting the result

y_pred_knn = knn.predict(x_test)

# checking the accuracy of the model
# creating the confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# jaccard score
js_knn = jaccard_score(y_test, y_pred_knn)

# f1_score 
f1_knn = f1_score(y_test, y_pred_knn)

# Descion tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 0)

dtc.fit(x_train, y_train)

# predicting result

y_pred_dtc = dtc.predict(x_test)

# creating confusion matrix
cm_dtc = confusion_matrix(y_test, y_pred_dtc)

# jaccard score

js_dtc = jaccard_score(y_test, y_pred_dtc)

# f1_score

f1_dtc = f1_score(y_test, y_pred_dtc)

#random_forest

from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(criterion = "entropy", random_state = 0)

rfc.fit(x_train, y_train)

# predicting the result
y_pred_rfc = rfc.predict(x_test)

# creating the confusion matrix
cm_rfc = confusion_matrix(y_test, y_pred_rfc)

# jaccard score
js_rfc = jaccard_score(y_test, y_pred_rfc)

# f1_score
f1_rfc = f1_score(y_test, y_pred_rfc)

from 


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.ensemble import GradientBoostingClassifier
xgb = GradientBoostingClassifier()

xgb.fit(x_train, y_train)

dropna()