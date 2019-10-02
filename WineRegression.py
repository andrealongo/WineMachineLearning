#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries 
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn import metrics


# In[2]:


#set wd
os.chdir('/Users/andrealongoni/Desktop/USIINFMScThesis')


# In[3]:


#load df and clean
df = pd.read_csv('WineClustering.csv')


# In[10]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['designation'])
df['designation'] = labelEncoder.transform(df['designation'])


# In[11]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['country'])
df['country'] = labelEncoder.transform(df['country'])


# In[12]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['province'])
df['province'] = labelEncoder.transform(df['province'])


# In[13]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['variety'])
df['variety'] = labelEncoder.transform(df['variety'])


# In[14]:


df.head()


# In[26]:


y = df['price']
X = df.loc[:, df.columns != 'price']


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[28]:


import statsmodels.api as sm


# In[29]:


model = sm.OLS(y,sm.add_constant(X)).fit()
# Output
print(model.summary())


# In[30]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[31]:


transformer = PolynomialFeatures(degree=2, include_bias=False)


# In[36]:


x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)


# In[37]:


model = LinearRegression().fit(x_, y)


# In[38]:


r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)


# In[40]:


## simplyfied df


# In[ ]:


df_short = df[['country', 'price', 'province', 'variety', 'vintage', 'designation']]


# In[43]:


y = df_short['price']
X = df_short.loc[:, df_short.columns != 'price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[44]:


model = sm.OLS(y,sm.add_constant(X)).fit()
# Output
print(model.summary())


# In[45]:


x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
model = LinearRegression().fit(x_, y)


# In[46]:


r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)


# In[ ]:




