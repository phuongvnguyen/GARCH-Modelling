#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{The Generalized Autoregressive Conditional Ceteroskedasticity (GARCH) Modelling }}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $\text{2. Dataset:}$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/ARCH-Modelling
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[38]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[7]:


data = pd.read_excel("data.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[8]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[16]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[17]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[18]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[19]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[25]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[26]:


daily_return.index


# ### Plotting returns

# In[33]:


fig=plt.figure(figsize=(10,7))
figure = daily_return['Return'].plot()


# In[30]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(daily_return.Return['2007':'2013'],LineWidth=4)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 31/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Return (%)',fontsize=17)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=17,fontweight='normal',color='k')


# # Modelling GARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 

# In[43]:


am = arch_model(daily_return,p=1, o=0, q=1)
res = am.fit(update_freq=1)
print(res.summary())


# # Checking the residual

# In[37]:



fig = res.plot(annualize='D')


# In[ ]:




