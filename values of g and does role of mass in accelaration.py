#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import csv 
import math, random


# In[22]:


def location(x_0,v_0,t):
    x=x_0+v_0*t-(9.8/2)*t**2
    
    return x


# In[32]:


with open('location.csv',mode='w') as gravity_file:
    gravity_writer=csv.writer(gravity_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    gravity_writer.writerow(['initial_position', 'initial_velocity', 'mass', 'time', 'location'])
    for i in range(1000):
        initial_position=random.randrange(1,10000)
        initial_velocity=random.randrange(1,100)
        mass=random.randrange(1,1000)
        time=random.randrange(1,100)
        gravity_writer.writerow([initial_position, initial_velocity, mass, time, location(initial_position, initial_velocity, time)])


# In[33]:


dataset=pd.read_csv("location.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[34]:


x.shape


# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

regressor=LinearRegression()
poly=PolynomialFeatures(degree=2)
x=poly.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[36]:


x.shape


# In[37]:


regressor.fit(x_train,y_train)


# In[38]:


coeff=regressor.coef_
b=regressor.intercept_


# In[39]:


coeff=np.around(coeff,decimals=2)


# In[40]:


coeff


# It can be seen that the coefficient corresponding to mass is 0,which indicates that the objects falling rate dosent depend on the mass of the object

# In[ ]:




