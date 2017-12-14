
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


# ## import data

# In[2]:


df = pd.read_csv('OnlineNewsPopularity.csv')
#dfx = df.loc[:, ' timedelta':' abs_title_sentiment_polarity']
#dfy = df.loc[:, ' shares']


# In[3]:


df = df.loc[:, ' timedelta':]
df.head()


# ## Exploratory Data Analysis (EDA)

# ### Feature distribution

# In[48]:


#datax = dfx.as_matrix(columns=None)
#datay = dfy.as_matrix(columns=None)

data = df.as_matrix(columns=None)


# In[6]:


#plot histogram
for i in range(0,data.shape[1]):
    plt.hist(data[:,i], bins='doane')  # arguments are passed to np.histogram
    plt.title("Histogram of feature "+str(i) + df.columns[i])
    plt.show()


# In[49]:


# remove outlier in var “n_unique_tokens”, “n_non_stop_words”, and “n_non_stop_unique_tokens”
data = data[data[:,3]<1]


# In[50]:


data.shape


# In[51]:


df.shape


# In[52]:


# eliminate missing value
c = np.array([10,19,43,44,45,47,48,49,52])
for i in c:
    data = data[data[:,i]!=0]
data.shape


# In[53]:


# reduce the skewness
# if the data contains no 0s, use log, else use sqrt

c = np.array([2,6,7,8,9,21,25,26,27,28,29,38,39,40,41,42,46])
for i in c:
    if (LA.norm(data[:,i],0)==data[:,i].shape):
        data[:,i] = np.log(data[:,i])
    else:
        data[:,i] = np.sqrt(data[:,i])


# In[40]:


# eliminate unvalid data
#c = np.array([18,20,22,24])
#data = np.delete(data, c, 1)
#data.shape


# ### Boolean feature effects

# In[83]:


labels = ['Lifestyle','Entertainment','Business','Social Media','Tech','World']
plot_set1 = data[data[:,12]==1][:,59] 
plot_set2 = data[data[:,13]==1][:,59]
plot_set3 = data[data[:,14]==1][:,59]
plot_set4 = data[data[:,15]==1][:,59]
plot_set5 = data[data[:,16]==1][:,59]
plot_set6 = data[data[:,17]==1][:,59]
plotdata = [plot_set1, plot_set2, plot_set3, plot_set4, plot_set5,plot_set6]
plt.boxplot(plotdata,labels=labels)
plt.xlabel('news category')
plt.ylabel('log_shares')
plt.title('Boxplot of category effect')
plt.tight_layout()
plt.show()


# In[84]:


labels = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun','Weekend']
plot_set1 = data[data[:,30]==1][:,59] 
plot_set2 = data[data[:,31]==1][:,59]
plot_set3 = data[data[:,32]==1][:,59]
plot_set4 = data[data[:,33]==1][:,59]
plot_set5 = data[data[:,34]==1][:,59]
plot_set6 = data[data[:,35]==1][:,59]
plot_set7 = data[data[:,36]==1][:,59]
plot_set8 = data[data[:,37]==1][:,59]
plotdata = [plot_set1, plot_set2, plot_set3, plot_set4, plot_set5,plot_set6,plot_set7,plot_set8]
plt.boxplot(plotdata,labels=labels)
plt.xlabel('news publish day')
plt.ylabel('log_shares')
plt.title('Boxplot of weekday effect')
plt.tight_layout()
plt.show()


# ### PCA 

# In[41]:


# pre eliminate unvalid and unuseful features
c = np.array([12,13,14,15,16,17,18,20,22,24,30,31,32,33,34,35,36,37])
data = np.delete(data, c, 1)
data.shape


# In[56]:


my_data = np.copy(data)
datax = my_data[:,:data.shape[1]-1]
datay = my_data[:,data.shape[1]-1]
for j in range(0,datay.shape[0]):
    if datay[j] >1400:
        datay[j] =1
    else:
        datay[j] =0


# In[57]:


datax_train = datax[:int(datax.shape[0]*3/4),:]
datay_train = datay[:int(datay.shape[0]*3/4)]

datax_test = datax[int(datax.shape[0]*3/4):,:]
datay_test = datay[int(datay.shape[0]*3/4):]


# In[58]:


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


# In[59]:


lr = LogisticRegression()
lr.fit(datax_train, datay_train) 
lr.score(datax_test, datay_test) 


# In[60]:


myscores


# In[159]:


my_data = np.copy(data)
datax = my_data[:,:59]
datay = my_data[:,59]
for j in range(0,datay.shape[0]):
    if datay[j] >1400:
        datay[j] =1
    else:
        datay[j] =0


# In[157]:


data.shape

