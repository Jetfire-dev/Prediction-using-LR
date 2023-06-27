#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ds2= pd.read_csv('cost_revenue_clean.csv')


# In[3]:


ds2


# In[4]:


plt.xlabel('production_budget_usd')
plt.ylabel('worldwide_gross_usd')
plt.scatter(ds2.production_budget_usd,ds2.worldwide_gross_usd,color='red',marker='+')


# In[5]:


x=ds2[['production_budget_usd']]
x


# In[6]:


y= ds2[['worldwide_gross_usd']]


# In[7]:


y


# In[9]:


reg=LinearRegression()


# In[10]:


reg.fit(x,y)


# In[11]:


y1=reg.coef_*x+reg.intercept_


# In[12]:


plt.figure(figsize=(6,6))
plt.xlabel('production_budget_usd')
plt.ylabel('worldwide_gross_usd')
plt.scatter(x,y,color='red')
plt.plot(x,y1,color='black')
plt.show()


# In[13]:


reg.intercept_


# In[14]:


reg.coef_


# In[15]:


reg.predict([[2500]])


# In[16]:


3.11150918*2500+-7236192.72913958


# In[17]:


reg.predict([[5000]])


# In[18]:


3.11150918*5000+-7236192.72913958


# In[19]:


plt.figure(figsize=(6,6))
plt.xlabel('production_budget_usd')
plt.ylabel('worldwide_gross_usd')
plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='black')
plt.show()


# In[20]:


reg.score(x,y)


# In[ ]:




