#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# In[2]:


train = pd.read_csv('hw3_Data2/train.csv')
test = pd.read_csv('hw3_Data2/test.csv')

train = pd.DataFrame(train,columns=['Date','Close'])
test = pd.DataFrame(test,columns=['Date','Close'])

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

train.set_index('Date',inplace=True)
test.set_index('Date',inplace=True)

plt.plot(train)
plt.plot(test)


# ## use auto-arima

# In[12]:


#Build Model
auto_arima_model = pm.auto_arima(train,seasonal=False)
print(auto_arima_model.summary())


# In[13]:


#Predict
auto_arima_prediction = pd.DataFrame(auto_arima_model.predict(n_periods=41),index = test.index)
auto_arima_prediction.columns = ['prediction']


# In[14]:


#Plot the result
#plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='train')
plt.plot(test, label='test')
plt.plot(auto_arima_prediction, label='forecast')
plt.xlabel('date')
plt.ylabel('close value')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.savefig('part3.png')
plt.show()


# ## use arima when order = (0,1,0)

# In[15]:


# Build Model
#model = ARIMA(train, order=(3,2,1))
arima_model = ARIMA(train, order=(0, 1, 0))  
fitted = arima_model.fit(disp=-1)  


# In[16]:


# Forecast
n_periods = 41
fc, se, conf = fitted.forecast(n_periods)
arima_prediction = pd.Series(fc, index=test.index)


# In[17]:


#Plot the result
#plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='train')
plt.plot(test, label='test')
plt.plot(arima_prediction, label='forecast')
plt.xlabel('date')
plt.ylabel('close value')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.savefig('part3.png')
plt.show()


# ## use ARIMA when seasonal = True

# In[18]:


auto_sarima_model = pm.auto_arima(train,m=12,seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
auto_sarima_model.summary()


# In[19]:


#Predict
auto_sarima_prediction = pd.DataFrame(auto_sarima_model.predict(n_periods=41),index = test.index)
auto_sarima_prediction.columns = ['prediction']


# In[20]:


#Plot the result
#plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='train')
plt.plot(test, label='test')
plt.plot(auto_sarima_prediction, label='forecast')
plt.xlabel('date')
plt.ylabel('close value')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.savefig('part3.png')
plt.show()


# ## use ARIMAX when order = (0,1,0),seasonal_order=(2,1,0,12)

# In[21]:


#smodel = SARIMAX(train, trend='n', order=(0,1,0), seasonal_order=(2,1,0,12))
sarima_model = SARIMAX(train, order=(0,1,0), seasonal_order=(2,1,0,12))
fitted = sarima_model.fit(disp=-1)
#fitted = smodel.fit()
print (fitted.summary())


# In[22]:


# Forecast
n_periods = 41
sarima_prediction = fitted.forecast(n_periods,index = test.index)


# In[23]:


#Plot the result
#plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='train')
plt.plot(test, label='test')
plt.plot(sarima_prediction, label='forecast')
plt.xlabel('date')
plt.ylabel('close value')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.savefig('part3.png')
plt.show()


# ## count MSE

# In[24]:


mean_squared_error(test, auto_arima_prediction)


# In[25]:


mean_squared_error(test, arima_prediction)


# In[26]:


mean_squared_error(test, auto_sarima_prediction)


# In[27]:


mean_squared_error(test, sarima_prediction)


# In[ ]:




