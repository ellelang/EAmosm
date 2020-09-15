#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import sqlalchemy
import pandas as pd 
import numpy as np 
import seaborn as sns 
import ast
import re
from matplotlib import pyplot as plt
import sys
from scipy.stats import norm
import math
import random
import time
from datetime import timedelta


# In[4]:


segdf_cmax = pd.read_csv ('data_2019/cmax_detarget1.csv')
segdf_hbo = pd.read_csv ('data_2019/hbo_detarget.csv')


# In[5]:


segdf_cmax.shape


# In[19]:


cand_cmax = pd.read_csv ('data_2019/cmax_cand.csv')
cand_hbo = pd.read_csv ('data_2019/hbo_cand.csv')


# In[6]:


segdf_cmax.head(2)


# In[7]:


n = segdf_cmax.shape[0]


# In[20]:


desegid_cmax = segdf_cmax.base_seg_id.tolist()
desegid_hbo = segdf_hbo.base_seg_id.tolist()


# In[6]:


cand_cmax.head(2)


# In[13]:


rank_cand = cand_cmax.sort_values(by= 'cr').base_seg_id.tolist()


# In[21]:


len(list(set(desegid_hbo) & set(desegid_cmax)))


# In[17]:


"AND NOT (" + ' '.join("s={} OR".format(n) for n in desegid )


# In[ ]:




