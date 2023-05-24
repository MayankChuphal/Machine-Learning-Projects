#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd


# In[48]:


df=pd.read_csv('Placement.csv')


# In[49]:


df.head()


# In[50]:


df.shape


# In[51]:


df.info() ##no missing values here


# ### steps:-
# ### 0. Preprocess+ EDA+ Feature Selection
# ### 1. Extract input and output cols
# ### 2. Scale the values
# ### 3.Train test split
# ### 4.Train the model
# ### 5.Evaluate the model/model selection
# ### 6.Deploy the model

# ### 0.preprocess+EDA+Feature selection
# ### here there is no need of unnamed column so we will remove it

# In[52]:


### iloc:- is used to select rows and columns from a pandas dataframe 


# In[53]:


df=df.iloc[:,1:]   ##: includes all rows and 1: excludes 0 th column and includes others


# In[54]:


df.head()


# In[55]:


import matplotlib.pyplot as plt  ##edapart


# In[56]:


plt.scatter(df['cgpa'],df['iq'],c=df['placement'])


# In[57]:


## yellow indicates placement    ##violet indicates no placement


# In[58]:


## logistic regression can be used here as our data is linear we can separate classes linearly


# In[59]:


### feature selection:no need here


# In[60]:


## independent variables: iq and cgpa and dependent :placement 


# ### 1. separate input ouput data

# In[61]:


x=df.iloc[:,0:2]
y=df.iloc[:,-1]


# In[62]:


x


# In[63]:


y


# ### 3. Train test and split (we are doing 3 rd step first here)

# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1) 

#10% data out of 100 will go to test and reamining 90 will go to training 


# In[66]:


x_train


# In[67]:


y_train


# In[68]:


x_test


# In[69]:


y_test


# ### 2.Scaling the data

# In[70]:


from sklearn.preprocessing import StandardScaler


# In[71]:


scaler =StandardScaler() ##creating object of this class


# In[72]:


x_train=scaler.fit_transform(x_train) ##it will understand the data and then transform the data


# In[73]:


x_train ##data is scaled from range of -1 to 1


# In[74]:


x_test=scaler.transform(x_test) ##no need of fit here as pattern is already recognised earlier in x_train


# In[75]:


x_test


# ### 4.train the model

# In[76]:


from sklearn.linear_model import LogisticRegression


# In[77]:


clf=LogisticRegression()


# In[78]:


clf.fit(x_train,y_train) ##fit function is used to train data


# ### 5.model evaluation

# In[79]:


y_pred=clf.predict(x_test) ##predict the output of test data


# In[80]:


y_pred


# In[81]:


y_test


# In[82]:


from sklearn.metrics import accuracy_score


# In[83]:


accuracy_score(y_test,y_pred) ##this will give the accuracy score


# ### how to plot decision boundary(what pattern ml model noticed in
# ### data can be visualized using decison boundary)

# In[84]:


from mlxtend.plotting import plot_decision_regions


# In[85]:


plot_decision_regions(x_train, y_train.values, clf=clf, legend=2) ##.values is used to convert into arrayformat


# In[86]:


import pickle


# In[87]:


## convert the object into file


# In[88]:


pickle.dump(clf,open('model.pkl','wb'))


# In[ ]:





# In[ ]:




