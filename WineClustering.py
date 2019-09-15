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

df = df[['country', 'price', 'province', 'variety', 'vintage', 'quality']]

#df = df.drop(['Unnamed: 0', 'description', 'title', 'winery', 'sentiment'], axis=1)


# In[4]:


df.head()


# In[25]:


prov = df.variety.value_counts()


# In[27]:


len(prov)


# In[26]:


prov


# ## Clustering

# In[13]:


# create blobs
data = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=1.6, random_state=50)
# create np array for data points
points = data[0]
# create scatter plot
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
plt.xlim(-15,15)
plt.ylim(-15,15)


# In[14]:


print("***** Check NAs *****")
print(df.isna().sum())


# ## a little bit of data visualization (non categorical values against quality)

# In[15]:


g = sns.FacetGrid(df, col='quality')
g.map(plt.hist, 'vintage', bins=20)


# In[16]:


g = sns.FacetGrid(df, col='quality')
g.map(plt.hist, 'price', bins=100)


# In[31]:


df.info()


# In[47]:


#ecoding categorical values
labelEncoder = LabelEncoder()
#labelEncoder.fit(df['country'])
#labelEncoder.fit(df['province'])
#labelEncoder.fit(df['variety'])
labelEncoder.fit(df['quality'])
#df['country'] = labelEncoder.transform(df['country'])
#df['province'] = labelEncoder.transform(df['province'])
#df['variety'] = labelEncoder.transform(df['variety'])
df['quality'] = labelEncoder.transform(df['quality'])


# In[65]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['country'])
df['country'] = labelEncoder.transform(df['country'])


# In[66]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['province'])
df['province'] = labelEncoder.transform(df['province'])


# In[67]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['variety'])
df['variety'] = labelEncoder.transform(df['variety'])


# In[49]:


labelEncoder = LabelEncoder()
labelEncoder.fit(df['designation'])
df['designation'] = labelEncoder.transform(df['designation'])


# # SUPERVISED MACHINE LEARNING

# ## Naive bayes

# In[10]:


df.head()


# In[68]:


#since it's clasification jsut 0,1 0 = high quality and 1 = medium and low quality
df['quality'][df['quality'] == 2]=1


# In[69]:


#first drop the quality column 
X = np.array(df.drop(['quality'], 1).astype(object))
y = np.array(df['quality'])


# In[70]:


#split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[37]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


# In[38]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[39]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[40]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy Näive beyes:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))


# In[41]:


X_test[1,]


# In[42]:


df_names = pd.read_csv('WineClustering.csv')
df_names = df_names[['country', 'price', 'province', 'variety', 'vintage', 'quality','title', 'description']]


# In[44]:


y_pred_spec = gnb.predict(X_test)


# In[45]:


y_pred_spec = pd.DataFrame(y_pred_spec)


# In[46]:


final = pd.concat([df_names, y_pred_spec],axis = 1)


# In[47]:


final.head()


# In[48]:


final.columns = ['country', 'price', 'province', 'variety', 
                'vintage', 'quality', 'title', 'description', 'prediction']


# In[49]:


#print where quality is either low or medium and prediction was high (0)
df_toprint = final[(final.quality == 1) & (final.prediction == 0.0)]
df_toprint


# ## k-NN

# In[54]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[55]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[56]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[57]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy k-NN:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))


# ## Logistic regression

# In[87]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs')

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# In[88]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[99]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[100]:


print("Accuracy Logistic:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))


# In[91]:


y_pred_final = logreg.predict(X_test)
y_pred_final = pd.DataFrame(y_pred_final)


# In[92]:


final = pd.concat([df_names, y_pred_final],axis = 1)
final.columns = ['country', 'price', 'province', 'variety', 
                'vintage', 'quality', 'title', 'description', 'prediction']


# In[93]:


#print where quality is either low or medium and prediction was high (0)
df_toprint = final[(final.quality == 1) & (final.prediction == 0.0)]
df_toprint


# In[98]:


len(df_toprint)


# # UNSUPERVISED MACHINE LEARNING

# ## K-MEANS

# ## we start with clastering  ;) with K-means model

# In[51]:


#first drop the quality column 
X = np.array(df.drop(['quality'], 1).astype(object))


# In[13]:


X


# In[52]:


y = np.array(df['quality'])


# In[61]:


y


# In[53]:


#split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[79]:


kmeans = KMeans(n_clusters=2) # Cluster the wine records into 3: low, medium and high quality
kmeans.fit(X)


# In[80]:


#let's see how many wine were correctly clustered 
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[77]:


#let's tune our algorithm
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)


# In[78]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[81]:


#take 0 - 1 as the uniform value range across all the features.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[84]:


print(kmeans.fit(X_scaled))
print('Accuracy k-Means after tuning and normalization: 0.093')
print('Accuracy k-Means after tuning: 0.6334')
print('Accuracy k-Means: 0.3665')


# In[83]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# ## Hierachical and agglomerative clustering

# In[70]:


# notice: we can only take two value to plot the dendogram


# In[71]:


#Hierachical


# In[207]:


import scipy.cluster.hierarchy as shc
X_dendo = X_scaled[:, [3, 4]]
dendrogram = shc.dendrogram(shc.linkage(X_dendo, method='ward'))


# In[ ]:


#Agglomerative


# In[208]:


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(X_dendo)
labels = model.labels_


# In[209]:


plt.scatter(X_dendo[labels==0, 0], X_dendo[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X_dendo[labels==1, 0], X_dendo[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X_dendo[labels==2, 0], X_dendo[labels==2, 1], s=50, marker='o', color='green')
plt.show()


# In[216]:


df.head()


# ## PCA

# In[86]:


from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(df)


# In[87]:


existing_2d = pca.transform(df)


# In[95]:


existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = df.index
existing_df_2d.columns = ['PC1','PC2','PC3','PC4','PC5']
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(existing_df_2d.head())
print('Variance explained:')
print(pca.explained_variance_ratio_) 
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")


# In[89]:


print(pca.explained_variance_ratio_) 


# In[ ]:





# In[222]:


get_ipython().run_line_magic('matplotlib', 'inline')

ax = existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))

for i, country in enumerate(df.index):
    ax.annotate(country, (existing_df_2d.iloc[i].PC2, existing_df_2d.iloc[i].PC1))


# In[224]:


#Let's now create a bubble chart, by setting the point size to a value proportional 
#to the mean value for all the years in that particular country. 
#First we need to add a new column containing the re-scaled mean per country across all the years.
from sklearn.preprocessing import normalize

existing_df_2d['country_mean'] = pd.Series(df.mean(axis=1), index=existing_df_2d.index)
country_mean_max = existing_df_2d['country_mean'].max()
country_mean_min = existing_df_2d['country_mean'].min()
country_mean_scaled = (existing_df_2d.country_mean-country_mean_min) / country_mean_max
existing_df_2d['country_mean_scaled'] = pd.Series(
    country_mean_scaled, 
    index=existing_df_2d.index)
existing_df_2d.head()


# In[225]:


#Now we are ready to plot using this variable size (we will ommit the country names this time since we are not so interested in them).
existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', s=existing_df_2d['country_mean_scaled']*100, figsize=(16,8))


# In[227]:


#Let's do the same with the sum instead of the mean.
existing_df_2d['country_sum'] = pd.Series(df.sum(axis=1), index=existing_df_2d.index)
country_sum_max = existing_df_2d['country_sum'].max()
country_sum_min = existing_df_2d['country_sum'].min()
country_sum_scaled = (existing_df_2d.country_sum-country_sum_min) / country_sum_max
existing_df_2d['country_sum_scaled'] = pd.Series(
    country_sum_scaled, 
    index=existing_df_2d.index)
existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', s=existing_df_2d['country_sum_scaled']*100, figsize=(16,8))


# In[230]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6)

clusters = kmeans.fit(df)


# In[231]:


existing_df_2d['cluster'] = pd.Series(clusters.labels_, index=existing_df_2d.index)


# In[233]:


import numpy as np

axk =existing_df_2d.plot(
    kind='scatter',
    x='PC2',y='PC1',
    c=existing_df_2d.cluster.astype(np.float), 
    figsize=(16,8))

for i, country in enumerate(df.index):
    axk.annotate(country, (existing_df_2d.iloc[i].PC2 + 2, existing_df_2d.iloc[i].PC1 + 2))


# In[ ]:




