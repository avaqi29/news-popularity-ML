
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from numpy import genfromtxt
from numpy import linalg as LA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from numpy import ma

from sklearn.ensemble import RandomForestClassifier

import graphviz
from sklearn import tree


# In[2]:


df = pd.read_csv('OnlineNewsPopularity.csv')
dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
dfy = df.loc[:, ' shares']

datax = dfx.as_matrix(columns=None)
datay = dfy.as_matrix(columns=None)

for j in range(0,datay.shape[0]):
    if datay[j] >1400:
        datay[j] =1
    else:
        datay[j] =0

datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[3]:


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(datax_train, datay_train)
clf.score(datax_test,datay_test)


# In[4]:


clf = tree.DecisionTreeClassifier()
clf.fit(datax_train, datay_train)
clf.score(datax_test,datay_test)


# In[13]:


clf.tree_.node_count


# In[16]:


#import pydotplus
features = dfx.columns


# In[18]:


dot_data = tree.export_graphviz(clf.estimators_[0], out_file=None,feature_names=features,class_names='popularity',  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("tree01forallfeatures_maxd2,rs0")


# In[6]:


dfx.columns

