#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sweetviz as sw
import seaborn as sns
sns.set()


# In[5]:


data=pd.read_csv('creditcard.csv')
data=data.drop(['Time'],axis=1)
data['Class']=data['Class'].map({1:-1,0:1})
data.head()


# In[6]:


from sklearn.model_selection import train_test_split
train,test,=train_test_split(data,test_size=(0.1),random_state=42)
train,val=train_test_split(train,test_size=(0.1),random_state=43)


# In[7]:


train_normal=train[train['Class']==1].drop(['Class'],axis=1)
train_outlier=train[train['Class']==-1]


# In[8]:


from sklearn.svm import OneClassSVM 
outlier_prop = len(train_outlier) / len(train_normal) 
svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.1) 
svm.fit(train_normal)


# In[9]:


#train set
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(train.iloc[:,-1],svm.predict(train.iloc[:,:-1]))
print(cm)
plt.matshow(cm,cmap=plt.cm.gray)


# In[10]:


#validation set
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(val.iloc[:,-1],svm.predict(val.iloc[:,:-1]))
print(cm)
plt.matshow(cm,cmap=plt.cm.gray)


# In[ ]:


#train set
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(train.iloc[:,-1],svm.predict(train.iloc[:,:-1])))
print(recall_score(train.iloc[:,-1],svm.predict(train.iloc[:,:-1])))
print(f1_score(train.iloc[:,-1],svm.predict(train.iloc[:,:-1])))


# In[43]:


#validation set
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(val.iloc[:,-1],svm.predict(val.iloc[:,:-1])))
print(recall_score(val.iloc[:,-1],svm.predict(val.iloc[:,:-1])))
print(f1_score(val.iloc[:,-1],svm.predict(val.iloc[:,:-1])))


# In[44]:


#train set
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(train.iloc[:,-1],svm.predict(train.iloc[:,:-1]))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,0],[1,1],'k--')
print(roc_auc_score(train.iloc[:,-1],svm.predict(train.iloc[:,:-1])))


# In[45]:


#validation set
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(val.iloc[:,-1],svm.predict(val.iloc[:,:-1]))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,0],[1,1],'k--')
print(roc_auc_score(val.iloc[:,-1],svm.predict(val.iloc[:,:-1])))


# In[4]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X=add_constant(data.iloc[:,:-2])
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[ ]:




