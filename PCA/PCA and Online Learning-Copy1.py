
# coding: utf-8

# # PCA and online learning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from numpy import genfromtxt
from numpy import linalg as LA


# In[2]:


df = pd.read_csv('OnlineNewsPopularity.csv')
dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
dfy = df.loc[:, ' shares']

datax = dfx.as_matrix(columns=None)
datay = dfy.as_matrix(columns=None)


# In[3]:


datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[4]:


datax_train_cov = np.cov(datax_train,rowvar=False)
datax_train_cov


# In[5]:


data_eigvals, data_eigvecs = np.linalg.eig(datax_train_cov)
data_eigvals


# In[6]:


data_eigvecs


# In[7]:


nums = np.arange(1,data_eigvals.size+1)
plt.plot(nums, np.sort(data_eigvals)[::-1])
plt.yscale('log')
plt.show()


# In[8]:


first_index = data_eigvals.argsort()[-1:][::-1]
last_index = data_eigvals.argsort()[-20:-19][::-1]
matrix_A = data_eigvecs[:,:20]
#matrix_A = np.transpose(matrix_A)
matrix_A


# In[9]:


data_PCA = np.dot(datax_train,matrix_A )
data_PCA.shape


# ## Online learning with static expert

# In[10]:


t=1
pt = np.ones(data_PCA.shape[1])/data_PCA.shape[1]
b=0.000000001
#b=0.000000001
steps = data_PCA.shape[0]
#steps = 20000
iterLoss = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(data_PCA[j,:] , pt))-datay_train[j])
    for i in range(0,data_PCA.shape[1]):
        loss = np.square(data_PCA[j,i]-datay_train[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)
lt = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    for i in range(0,j+1):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)


# In[11]:


plt.plot(lt)
plt.show()


# In[12]:


matrix_A = data_eigvecs[:,:10]
#matrix_A = np.transpose(matrix_A)
data_PCA = np.dot(datax_train,matrix_A )

t=1
pt = np.ones(data_PCA.shape[1])/data_PCA.shape[1]
b=0.000000001
#b=0.000000001
steps = data_PCA.shape[0]
#steps = 20000
iterLoss = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(data_PCA[j,:] , pt))-datay_train[j])
    for i in range(0,data_PCA.shape[1]):
        loss = np.square(data_PCA[j,i]-datay_train[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)
lt = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    for i in range(0,j+1):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)


# In[13]:


plt.plot(lt)
plt.show()


# In[14]:


matrix_A = data_eigvecs[:,:5]
#matrix_A = np.transpose(matrix_A)
data_PCA = np.dot(datax_train,matrix_A )

t=1
pt = np.ones(data_PCA.shape[1])/data_PCA.shape[1]
b=0.000000001
#b=0.000000001
steps = data_PCA.shape[0]
#steps = 20000
iterLoss = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(data_PCA[j,:] , pt))-datay_train[j])
    for i in range(0,data_PCA.shape[1]):
        loss = np.square(data_PCA[j,i]-datay_train[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)
lt = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    for i in range(0,j+1):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)


# In[15]:


plt.plot(lt)
plt.show()


# In[16]:


matrix_A = data_eigvecs[:,:2]
#matrix_A = np.transpose(matrix_A)
data_PCA = np.dot(datax_train,matrix_A )

t=1
pt = np.ones(data_PCA.shape[1])/data_PCA.shape[1]
b=0.000000001
#b=0.000000001
steps = data_PCA.shape[0]
#steps = 20000
iterLoss = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    iterLoss[j] = np.square(np.sum(np.multiply(data_PCA[j,:] , pt))-datay_train[j])
    for i in range(0,data_PCA.shape[1]):
        loss = np.square(data_PCA[j,i]-datay_train[j])
        pt[i] = pt[i]*np.power(np.exp(1),(-b * loss))
    pt = pt/np.sum(pt)
lt = np.zeros(data_PCA.shape[0])
for j in range(0,steps):
    if j == 0:
        lt[j] = iterLoss[0]
    for i in range(0,j):
        lt[j]+= iterLoss[i]
    lt[j]=lt[j]/(j+1)


# In[17]:


plt.plot(lt)
plt.show()


# In[18]:


lt


# In[19]:


lt[0]


# ## Bad attempt
