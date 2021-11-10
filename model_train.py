#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE NECESSARY MODULES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# # import the dataset

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/house price prediction/boston_house_dataset.csv')


# In[3]:


df.head()


# In[4]:


#shape of the dataframe
df.shape


# In[5]:


#checking for missing values
df.isnull().sum()


# In[6]:


df.info()


# In[7]:


#statistical measure of data
df.describe()


# In[8]:


df['PRICE'] = df['MEDV']


# In[9]:


df = df.drop('MEDV', axis = 1)


# In[10]:


#understanding the correlation between various features in the dataset
correlation = df.corr()


# In[11]:


#constructing a heatmap to understand the correlation
plt.figure(figsize = (10,10))
sns.heatmap(correlation, cbar = True, fmt = '.1f', square = True, annot = True, annot_kws = {'size' : 8}, cmap = 'Blues')


# In[14]:


#splitting the data and target
X = df.drop('PRICE', axis = 1)
Y = df['PRICE']


# In[15]:


print(X.shape)
print(Y.shape)


# # splitting the data into train and test data and model training
# 

# In[16]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[17]:


#model training
model = XGBRegressor()


# In[18]:


#training the model on training data
model.fit(x_train, y_train)


# # prediction on training data

# In[19]:


#accuracy of prediction on training data
predict_values = model.predict(x_train)


# In[20]:


print(predict_values)


# In[21]:


#R squared error
score1 = metrics.r2_score(y_train, predict_values)

#mean absolute error
score2 = metrics.mean_absolute_error(y_train, predict_values)


# In[22]:


print('R square error :', score1)
print('mean square error :', score2)


# In[30]:


#visualize the actual price and the predicted price
plt.scatter(y_train, predict_values, color = 'red')
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')


# # predicting the values on test data

# In[23]:


#accuracy of prediction on testing data
test_predict_values = model.predict(x_test)


# In[24]:


#R squared error
score1 = metrics.r2_score(y_test, test_predict_values)

#mean absolute error
score2 = metrics.mean_absolute_error(y_test, test_predict_values)


# In[25]:


print('R square error :', score1)
print('mean square error :', score2)


# In[31]:


#visualize the actual price and the predicted price
plt.scatter(y_test, test_predict_values, color = 'red')
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL PRICE VS. PREDICTED PRICE')


# # model testing

# In[36]:


input_value = input()
input_array = [float(i) for i in input_value.split(',')]

input_array = np.asarray(input_array)
reshaped_array = input_array.reshape(1, -1)
#print(reshaped_array)
predict = model.predict(reshaped_array)
print('THE PREDICTED PRICE IS :', predict)


# In[ ]:




