
#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



# In[10]:



# In[11]:



# In[19]:


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('data.csv')


# In[21]:


# print first 5 rows of the dataset
heart_data.head()


# In[22]:


# print last 5 rows of the dataset
heart_data.tail()


# In[23]:


# number of rows and columns in the dataset
heart_data.shape


# In[24]:


# getting some info about the data
heart_data.info()


# In[25]:


# checking for missing values
heart_data.isnull().sum()


# In[26]:


# statistical measures about the data
heart_data.describe()


# In[27]:


# checking the distribution of Target Variable
heart_data['target'].value_counts()


# In[29]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


# In[30]:


print(X)


# In[31]:


print(Y)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[33]:


print(X.shape, X_train.shape, X_test.shape)


# In[41]:


model = LogisticRegression(max_iter=3000)


# In[49]:


# training the LogisticRegression model with Training data
model.fit(X_train.values, Y_train.values)


# In[43]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[44]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[45]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[46]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[50]:


joblib.dump(model, 'heart.joblib')



# In[ ]:



