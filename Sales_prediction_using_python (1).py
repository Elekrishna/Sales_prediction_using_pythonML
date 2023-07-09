#!/usr/bin/env python
# coding: utf-8

# # import necessary libaries;

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('Advertising.csv')


# In[3]:


data.head(25)


# In[4]:


data.tail(10)


# In[5]:


data.shape


# In[6]:


data.info


# In[7]:


data.describe


# In[8]:


data = data.drop(columns=["Unnamed: 0"])


# In[9]:


data


# In[10]:


plt.figure(figsize=[7,5])
sns.boxplot(data)
plt.show()


# In[11]:


sns.histplot(data['TV']);


# In[12]:


sns.displot(data['Radio']);


# In[13]:


sns.displot(data['Newspaper']);


# In[14]:


sns.histplot(data['Sales']);


# In[15]:


sns.pairplot(data,hue='Sales');


# # Logistic_Regrassion

# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=43)


# In[18]:


x = data.iloc[:,0:-1]


# In[19]:


x


# In[21]:


y = data.iloc[:, -1]


# In[22]:


y


# In[24]:


x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[34]:


logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[27]:





# # Prediction & Evaluation

# In[35]:


predictions = logmodel.predict(x_test)


# In[36]:


from sklearn.metrics import classification_report


# In[37]:


print(classification_report(y_test,predictions))


# In[ ]:




