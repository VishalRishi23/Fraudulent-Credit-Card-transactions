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


# In[2]:


data=pd.read_csv('creditcard.csv')
data.head()


# In[3]:


data['Amount'].describe()


# In[4]:


data['Class'].value_counts()


# In[5]:


sns.distplot(data['Amount'])


# In[6]:


data.Amount.quantile(0.9855)


# In[7]:


data=data[data['Amount']<=835.0]
data.head()


# In[8]:


sns.distplot(data['Time'])


# In[9]:


data['Time'].describe()


# In[10]:


data=data.drop(['Time'],axis=1)
data.head()


# In[11]:


plt.scatter(pow(data['Amount'],1),data['Class'])


# In[12]:


data['Class'].value_counts()


# In[13]:


report=sw.analyze(data)
report.show_html()


# In[14]:


Xtrain=data.iloc[:,:-1]
ytrain=data.iloc[:,-1]


# In[15]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(Xtrain,ytrain,test_size=(0.1),random_state=42)
Xtrain,Xval,ytrain,yval=train_test_split(Xtrain,ytrain,test_size=(0.1),random_state=43)


# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.utils import class_weight
#a=class_weight.compute_class_weight('balanced',np.array([0.,1.]),ytrain)
tree=DecisionTreeClassifier(max_depth=1,class_weight='balanced')
tree.fit(Xtrain,ytrain)
#pg={'max_depth':[1,2,3,4,5,6,7,8,9,10]}
#gs=GridSearchCV(tree,pg,cv=5,scoring='f1',refit=True,n_jobs=-1)
#gs.fit(Xtrain,ytrain)


# In[57]:


gs.best_params_


# In[17]:


#train set
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytrain,tree.predict(Xtrain))
print(cm)
plt.matshow(cm,cmap=plt.cm.gray)


# In[18]:


#validation set
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yval,tree.predict(Xval))
print(cm)
plt.matshow(cm,cmap=plt.cm.gray)


# In[19]:


#train set
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(ytrain,tree.predict(Xtrain)))
print(recall_score(ytrain,tree.predict(Xtrain)))
print(f1_score(ytrain,tree.predict(Xtrain)))


# In[20]:


#validation set
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(yval,tree.predict(Xval)))
print(recall_score(yval,tree.predict(Xval)))
print(f1_score(yval,tree.predict(Xval)))


# In[21]:


#train set
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(ytrain,tree.predict(Xtrain))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,0],[1,1],'k--')
print(roc_auc_score(ytrain,tree.predict(Xtrain)))


# In[22]:


#validation set
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(yval,tree.predict(Xval))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,0],[1,1],'k--')
print(roc_auc_score(yval,tree.predict(Xval)))


# In[37]:


#max_depth=1,2,3,7 produced best results, standout being 1
#recall=86.67% auc=92%


# In[ ]:




