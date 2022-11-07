#!/usr/bin/env python
# coding: utf-8

# I am writing code for survival predcition in Titanic using RandomForests and XGBoost methods.

# In[98]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv(r'C:\Users\ict\Downloads\python_small_projects\train.csv')
test=pd.read_csv(r'C:\Users\ict\Downloads\python_small_projects\test.csv')
#Data Visualization
print("train data\n",train.head())
print("test data\n",test.head())
print("train shape", train.shape)
print("test shape", test.shape)


# In[99]:


#Checking null values
train.isnull().sum()


# In[100]:


#Filling null values
for features in ['Age','Cabin', 'Embarked','Fare']:
    lbl = LabelEncoder()
    lbl.fit(pd.concat((train[features], test[features])) )
    train[features] = lbl.transform(train[features])
    test[features] = lbl.transform(test[features])
print(train.isnull().sum())
print(test.isnull().sum())
train.head()


# In[101]:


# Data mapping for string values
train["Sex"]=train["Sex"].map({"male":1, "female":0})
test["Sex"]=test["Sex"].map({"male":1, "female":0})
#seperating target value from train data
y_target = train['Survived']
#getting rid of unusefull data
train.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train.shape, test.shape


# In[102]:


#data preparaion for applying model(first 600 enteries in training and remaining in test data)
X_train=train.iloc[:600,:]
X_test=train.iloc[600:,:]
y_train=y_target.iloc[:600]
y_test=y_target.iloc[600:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[103]:


#Applying random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
model = RandomForestClassifier(n_estimators = 600, max_features=0.5)
model.fit(X_train, y_train)
predicted=model.predict(X_test)
print(accuracy_score(predicted, y_test))
Orig_test_predict=model.predict(test)


# In[104]:


#Applying XGBoost model
import xgboost as xgb
model1 = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.02)
model1.fit(X_train, y_train)
predicted1=model1.predict(X_test)
print(accuracy_score(predicted1, y_test))
#predicting oridinal test file
Orig_test_predict=model1.predict(test)


# In[133]:


Saving prediction in CSV file
test2=pd.read_csv(r'C:\Users\ict\Downloads\python_small_projects\test.csv')
talha_submission = pd.DataFrame({'PassengerID': test2["PassengerId"], 'Survived': Orig_test_predict})
talha_submission.to_csv(r'C:\Users\ict\Downloads\python_small_projects\predicted.csv', index=False)
pd.read_csv(r'C:\Users\ict\Downloads\python_small_projects\predicted.csv')

