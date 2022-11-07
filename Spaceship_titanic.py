#!/usr/bin/env python
# coding: utf-8

# In[70]:


#data Visualization
import pandas as pd
import numpy as np
train=pd.read_csv(r'C:\Users\ict\Downloads\python_projects\Spacehip Titanic\train.csv')
test=pd.read_csv(r'C:\Users\ict\Downloads\python_projects\Spacehip Titanic\test.csv')
train.head()


# In[72]:


print(train.shape)
print(train.isnull().sum())
print(test.shape)
print(test.isnull().sum())


# In[10]:


train.dtypes


# In[73]:


#imputing missing values in valueable columns
from sklearn.impute import SimpleImputer
imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]
imputer=SimpleImputer(strategy="median")
train[imputer_cols]=imputer.fit_transform(train[imputer_cols])
test[imputer_cols]=imputer.fit_transform(test[imputer_cols])
train.isnull().sum()


# In[139]:


#Labeling categorical values
from sklearn.preprocessing import LabelEncoder
labeler=LabelEncoder()
label_colms = ["PassengerId","HomePlanet", "CryoSleep","Cabin", "Destination" ,"VIP"]
for colms in label_colms:
    train[colms]=labeler.fit_transform(train[colms])
    test[colms]=labeler.fit_transform(test[colms])
train.head(5)


# In[82]:


#Data splitting in train_test data
target=train["Transported"]
droped_train=train.drop(['Transported','Name'], axis =1 )
X_train=droped_train.iloc[:6000,:]
y_train=target.iloc[:6000]
X_test=droped_train.iloc[6000:,:]
y_test=target.iloc[6000:]


# In[83]:


print(X_train.dtypes)


# In[137]:


#model applying
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=10, random_state=100)
model.fit(X_train, y_train)


# In[118]:


predicted=model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(predicted, y_test))


# In[135]:


# Checking confusion metrix and accuracy report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))


# In[95]:





# In[138]:


#Checking original test report
droped_test=test.drop(['Name'], axis =1 )
print(droped_test.head())
Orig_test_predict=model.predict(droped_test)


# In[134]:


#Saving prediction in CSV file
test2=pd.read_csv(r'C:\Users\ict\Downloads\python_projects\Spacehip Titanic\test.csv')
talha_submission = pd.DataFrame({'PassengerId': test2["PassengerId"], 'Transporated': Orig_test_predict})
talha_submission.to_csv(r'C:\Users\ict\Downloads\python_projects\spaceship_titanic.csv', index=False)
pd.read_csv(r'C:\Users\ict\Downloads\python_projects\spaceship_titanic.csv')

