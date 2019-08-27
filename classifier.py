# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:02:06 2019

@author: vinayver
"""

# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Importing the dataset
dataset = pd.read_csv('data/creditcard.csv')

# Exploring the dataset
print(dataset.columns)

print(dataset.shape)


# This dataset is very huge to explore and get the insights on data. Let's take a small sample 
# out of it.
new_df = dataset.sample(frac = 0.1, random_state =1 )
new_df.shape

# Plot histogram of each feature
new_df.hist(figsize=(12,4))
plt.show()

#Determine no of fraud cases in data
Fraud = new_df[new_df['Class'] == 1]
Valid = new_df[new_df['Class'] == 0]
outlier_frac = float(len(Fraud))/len(Valid)

print('outlier fraction : {}'.format(outlier_frac))
print('Valid Cases : {}'.format(len(Valid)))
print('Fraud Cases : {}'.format(len(Fraud)))
sns.countplot(new_df['Class'])

# Corelation Matrix
corr = new_df.corr()
fig = plt.figure(figsize=(12,8))
 
sns.heatmap(corr,vmax=0.8)
plt.show()

# get all the columns from dataframe
columns = new_df.columns.tolist()

# filter the dataset to remove data we donot want 
columns = [c for c in columns if c not in ['Class','Time']]

# Store the variable we will be predicting on
target = 'Class'

X = new_df[columns]
y = new_df[target]

# print the shapes of X and Y
X.shape
y.shape

# Our data is already scaled we should split our training and test sets
from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

       
# Classification without using any sampling techniques
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier       
from sklearn import metrics

model_lg = LogisticRegression()
model_lg.fit(X_train,y_train)
lg_pred = model_lg.predict(X_test)
metrics.accuracy_score(y_test,lg_pred)
metrics.confusion_matrix(y_test,lg_pred)

model_dt = DecisionTreeClassifier(max_depth=8)
model_dt.fit(X_train,y_train)
dt_pred = model_dt.predict(X_test)
metrics.accuracy_score(y_test,dt_pred)
metrics.confusion_matrix(y_test,dt_pred)


model_rf = RandomForestClassifier(max_depth=8)
model_rf.fit(X_train,y_train)
rf_pred = model_rf.predict(X_test)
metrics.accuracy_score(y_test,rf_pred)
metrics.confusion_matrix(y_test,rf_pred)


# Random Undersampling using SMOTE
from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler()
X_train_ros,y_train_ros = ros.fit_sample(X_train,y_train)

model_lg = LogisticRegression()
model_lg.fit(X_train_ros,y_train_ros)
lg_pred = model_lg.predict(X_test)
metrics.accuracy_score(y_test,lg_pred)
metrics.confusion_matrix(y_test,lg_pred)

model_dt = DecisionTreeClassifier(max_depth=8)
model_dt.fit(X_train_ros,y_train_ros)
dt_pred = model_dt.predict(X_test)
metrics.accuracy_score(y_test,dt_pred)
metrics.confusion_matrix(y_test,dt_pred)


model_rf = RandomForestClassifier(max_depth=8)
model_rf.fit(X_train_ros,y_train_ros)
rf_pred = model_rf.predict(X_test)
metrics.accuracy_score(y_test,rf_pred)
metrics.confusion_matrix(y_test,rf_pred)