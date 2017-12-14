
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
df = df.loc[:, ' timedelta':]
dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
dfy = df.loc[:, ' shares']

data = df.as_matrix(columns=None)

data = data[data[:,3]<1]

c = np.array([10,19,43,44,45,47,48,49,52])
for i in c:
    data = data[data[:,i]!=0]

'''
c = np.array([2,6,7,8,9,21,25,26,27,28,29,38,39,40,41,42,46])
for i in c:
    if (LA.norm(data[:,i],0)==data[:,i].shape):
        data[:,i] = np.log(data[:,i])
    else:
        data[:,i] = np.sqrt(data[:,i])
'''
# eliminate unvalid data
#c = np.array([18,20,22,24])
#data = np.delete(data, c, 1)

# pre eliminate unvalid and unuseful features
#c = np.array([12,13,14,15,16,17,18,20,22,24,30,31,32,33,34,35,36,37])
#data = np.delete(data, c, 1)

my_data = np.copy(data)
datax = my_data[:,:my_data.shape[1]-1]
datay = my_data[:,my_data.shape[1]-1]
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


myscorerf = 0.5
var_i =0
var_j=0
var_k=0
for i in range(5,25):
    for j in range(2,10):
        for k in range(0,2):
            clf = RandomForestClassifier(n_estimators=i, criterion='entropy', max_depth=j, random_state=k)
            clf.fit(datax_train, datay_train)
            temp = clf.score(datax_test,datay_test)
            if temp> myscorerf:
                myscorerf = temp
                var_i=i
                var_j=j
                var_k=k
myscorerf


# In[8]:


clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=7, random_state=0)
clf.fit(datax_train, datay_train)
clf.score(datax_test,datay_test)


# In[10]:


for i in range(5,25):
    clf = RandomForestClassifier(n_estimators=i, criterion='entropy', max_depth=7, random_state=0)
    clf.fit(datax_train, datay_train)
    temp = clf.score(datax_test,datay_test)
    plt.plot(i,temp,'ro',c='gold',markersize = 3)
plt.show();


# In[11]:


for i in range(2,10):
    clf = RandomForestClassifier(n_estimators=17, criterion='entropy', max_depth=i, random_state=0)
    clf.fit(datax_train, datay_train)
    temp = clf.score(datax_test,datay_test)
    plt.plot(i,temp,'ro',c='gold',markersize = 3)
plt.show();


# In[5]:


var_i


# In[6]:


var_j


# In[7]:


var_k


# In[3]:


clf = RandomForestClassifier(n_estimators=17, criterion='entropy', max_depth=7, random_state=0)
clf.fit(datax_train, datay_train)
clf.score(datax_test,datay_test)


# In[298]:


features = dfx.columns
targetnames = ['unpopular','popular']
dot_data = tree.export_graphviz(clf.estimators_[2], out_file=None,feature_names=features,class_names=targetnames,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("randomforest_20_7")


# In[107]:


targetnames = ['popular','nonpopular']


# In[270]:


clf.estimators_


# In[4]:


clf.feature_importances_


# In[6]:


vip_features = np.zeros([int(LA.norm(clf.feature_importances_,0)),2])
j=0
for i in range (0,clf.feature_importances_.shape[0]):
    if clf.feature_importances_[i]!=0:
        vip_features[j,0] = i
        vip_features[j,1] = clf.feature_importances_[i]
        j+=1


# In[7]:


clf.feature_importances_.shape


# In[8]:


vip_features


# ## Logistic regression with random forest features

# In[9]:


my_data = np.copy(data)
datax = my_data[:,vip_features[:,0].astype(int)]
datay = my_data[:,my_data.shape[1]-1]
for j in range(0,datay.shape[0]):
    if datay[j] >1400:
        datay[j] =1
    else:
        datay[j] =0
        
datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[10]:


clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(datax_train, datay_train)
clf.score(datax_test,datay_test)


# In[11]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(datax_train, datay_train) 
lr.score(datax_test, datay_test) 


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
myscores = np.zeros(datax_train.shape[1])
for i in range(1,datax_train.shape[1]+1):
    pca = PCA(n_components=i) 
    lr = LogisticRegression() 
    X_train_pca = pca.fit_transform(datax_train) 
    X_test_pca = pca.transform(datax_test) 
    lr.fit(X_train_pca, datay_train) 
    score = lr.score(X_test_pca, datay_test) 
    myscores[i-1] = score 
    plt.plot(i,score,'ro',c='gold',markersize = 3)
plt.show()


# In[16]:


myscores


# ## Try online learning

# In[176]:


t=1
pt = np.ones(datax.shape[1])/datax.shape[1]
b=0.0000000001
steps = datax.shape[0]
steps = 25000
iterLoss = np.zeros(datax.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(datax[j,:] , pt))-datay[j])
    for i in range(0,datax.shape[1]):
        loss = np.square(datax[j,i]-datay[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)

lt = np.zeros(datax.shape[0])
for j in range(0,steps):
    for i in range(0,j):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)
    plt.plot(j,lt[j],'ro',c='r',markersize = 1)
plt.show()


# In[177]:


lt[j]


# In[200]:


t=1
pt = np.ones(datax.shape[1])/datax.shape[1]
b=0.1
steps = datax.shape[0]
steps = 25000
iterLoss = np.zeros(datax.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(datax[j,:] , pt))-datay[j])
    for i in range(0,datax.shape[1]):
        loss = np.square(datax[j,i]-datay[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)

lt = np.zeros(datax.shape[0])
for j in range(0,steps):
    for i in range(0,j):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)
    plt.plot(j,lt[j],'ro',c='r',markersize = 1)
plt.show()


# In[201]:


lt[j]


# In[202]:


np.dot(datax[2500,:],pt)


# In[210]:


matches = 0
for i in range (datax_test.shape[0]):
    y_pred = np.dot(datax[i,:],pt)
    if y_pred <2:
        y_pred =0
    else:
        y_pred = 1
    if y_pred == datay_test[i]:
        matches+=1
matches/datax_test.shape[0]


# ## Regression

# In[238]:


df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.loc[:, ' timedelta':]
dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
dfy = df.loc[:, ' shares']

data = df.as_matrix(columns=None)

data = data[data[:,3]<1]

c = np.array([10,19,43,44,45,47,48,49,52])
for i in c:
    data = data[data[:,i]!=0]

c = np.array([2,6,7,8,9,21,25,26,27,28,29,38,39,40,41,42,46,59])
for i in c:
    if (LA.norm(data[:,i],0)==data[:,i].shape):
        data[:,i] = np.log(data[:,i])
    else:
        data[:,i] = np.sqrt(data[:,i])
# eliminate unvalid data
#c = np.array([18,20,22,24])
#data = np.delete(data, c, 1)

# pre eliminate unvalid and unuseful features
#c = np.array([12,13,14,15,16,17,18,20,22,24,30,31,32,33,34,35,36,37])
#data = np.delete(data, c, 1)

my_data = np.copy(data)
datax = my_data[:,vip_features[:,0].astype(int)]
datay = my_data[:,my_data.shape[1]-1]
        
datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[220]:


t=1
pt = np.ones(datax_train.shape[1])/datax_train.shape[1]
b=0.1
steps = datax_train.shape[0]
steps = 7000
iterLoss = np.zeros(datax_train.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(datax_train[j,:] , pt))-datay_train[j])
    for i in range(0,datax_train.shape[1]):
        loss = np.square(datax_train[j,i]-datay_train[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)

lt = np.zeros(datax_train.shape[0])
for j in range(0,steps):
    for i in range(0,j):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)
    plt.plot(j,lt[j],'ro',c='r',markersize = 1)
plt.show()


# In[221]:


lt[j]


# In[224]:


matches = 0
for i in range (datax_test.shape[0]):
    y_pred = np.dot(datax_test[i,:],pt)
    if y_pred <np.log(1400):
        y_pred =0
    else:
        y_pred = 1
    y_test = 1
    if datay_test[i] <np.log(1400):
        y_test =0
    if y_pred == y_test:
        matches+=1
matches/datax_test.shape[0]


# In[217]:


np.log(3)


# In[223]:


pt


# ## linear regression

# In[258]:


df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.loc[:, ' timedelta':]
dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
dfy = df.loc[:, ' shares']

data = df.as_matrix(columns=None)

data = data[data[:,3]<1]

c = np.array([10,19,43,44,45,47,48,49,52])
for i in c:
    data = data[data[:,i]!=0]

c = np.array([2,6,7,8,9,21,25,26,27,28,29,38,39,40,41,42,46,59])
for i in c:
    if (LA.norm(data[:,i],0)==data[:,i].shape):
        data[:,i] = np.log(data[:,i])
    else:
        data[:,i] = np.sqrt(data[:,i])

# eliminate unvalid data
c = np.array([18,20,22,24])
data = np.delete(data, c, 1)

# pre eliminate unvalid and unuseful features
c = np.array([12,13,14,15,16,17,18,20,22,24,30,31,32,33,34,35,36,37])
data = np.delete(data, c, 1)

my_data = np.copy(data)
datax = my_data[:,:my_data.shape[1]-1]
datay = my_data[:,my_data.shape[1]-1]
        
datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[259]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(datax_train,datay_train)
lr.score(datax_test, datay_test)


# In[262]:


matches = 0
for i in range (datax_test.shape[0]):
    y_pred = np.dot(datax_test[i,:],lr.coef_)
    if y_pred <np.log(1400):
        y_pred =0
    else:
        y_pred = 1
    y_test = 1
    if datay_test[i] <np.log(1400):
        y_test =0
    if y_pred == y_test:
        matches+=1
matches/datax_test.shape[0]


# In[261]:


lr = linear_model.Ridge(alpha=1)
lr.fit(datax_train,datay_train)
lr.score(datax_test, datay_test)

