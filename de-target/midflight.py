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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from collections import defaultdict
from functools import reduce
from collections import Counter
redshift_endpoint1 = "device-advertising.cqporcaaztp8.us-east-1.redshift.amazonaws.com"
redshift_user1 = "zhenlang"
redshift_pass1 = "PizzaTaco1"
port1 = 8192
dbname1 = "deviceadvertising"
from sqlalchemy import create_engine
from sqlalchemy import text
engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" % (redshift_user1, redshift_pass1, redshift_endpoint1, port1, dbname1)
engine1 = create_engine(engine_string)


# In[2]:



metricdata = pd.read_sql_query('''

with ads_data as (
select 
st.base_seg_id,
st.dim_date,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
sum(impressions) as impressions,
sum(clicks) as clicks,
case when sum(nvl(subscription_free_trial_click_purchases,0))+ sum(nvl(subscription_free_trial_view_purchases,0))
    +SUM(nvl(subscription_win_back_view_purchases, 0)
      + nvl(subscription_initial_promotion_view_purchases, 0)
      + nvl(subscription_win_back_click_purchases, 0)
      + nvl(subscription_initial_promotion_click_purchases, 0))>0 then sum(nvl(subscription_free_trial_click_purchases,0))+ sum(nvl(subscription_free_trial_view_purchases,0))
    +SUM(nvl(subscription_win_back_view_purchases, 0)
      + nvl(subscription_initial_promotion_view_purchases, 0)
      + nvl(subscription_win_back_click_purchases, 0)
      + nvl(subscription_initial_promotion_click_purchases, 0))

 when sum(nvl(amazon_pay_initial_click_purchases,0))+ sum(nvl(amazon_pay_initial_view_purchases,0)) > 0 then sum(nvl(amazon_pay_initial_click_purchases,0))+ sum(nvl(amazon_pay_initial_view_purchases,0)) end
 as conversions

 from
VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC st
LEFT JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_CONV sc
ON st.base_seg_id = sc.base_seg_id
and st.dim_date= sc.dim_Date
and st.marketplace_id=sc.marketplace_id
and st.ad_cfid = sc.ad_cfid
and st.targ_method = sc.targ_method
and st.seg_name = sc.seg_name

where st.targ_method = 'untargeted'  -- Filter on only untargeted segments
AND st.advertiser_name = 'HBO - Television - US'

group by 
st.base_seg_id,
st.dim_date,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method
)



select 
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined,
sum(st.impressions) as impressions,
sum(clicks) as clicks,
sum(conversions) as conversions,
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion
--nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and conversions > 0
    and a.cfid IN (9734794310501, 7557170310401, 5876111200201, 3692412370701, 2352684890701, 3226745300601, 3133083920101, 2230898560801, 7851697750801, 5720813520701, 8594754170501)

group by  
st.base_seg_id,
st.marketplace_id,
st.ad_cfid,
st.campaign_cfid,
st.start_dt_utc,
st.end_dt_utc,
st.dim_advertiser_id,
st.advertiser_name,
st.campaign_name,
st.entity_id,
st.seg_name,
st.targ_method,
property_type,
targeting_combined''', engine1)


# In[3]:


metricdata.shape


# In[4]:


metricdata.apply(pd.Series.nunique)


# In[5]:


def RuleFun (df, f):
    coldf = df.groupby('ad_cfid').agg(f).reset_index()
    coldf.columns = coldf.columns.droplevel(0)
    newsummary = coldf.rename(columns={'':'ad_cfid', '<lambda_0>': 'threshold'})
    mid_df = pd.merge(df, newsummary, how = 'left', left_on='ad_cfid', right_on = 'ad_cfid')
    mid_df['deviation'] = abs(mid_df['cost_per_conversion'] - mid_df['mean'])/mid_df['std']
    return mid_df

def DetargFun (df, col, method_name):
    df['mask'] = df[col] > df['threshold']
    
    f = {'base_seg_id': 'count','cost_per_conversion': 'mean', 'impressions': 'sum'}
    df_origin = df.groupby('ad_cfid').agg(f).reset_index()
    df_remain = df[df['mask'] == False].groupby('ad_cfid').agg(f).reset_index()
    df_detarg = df[df['mask'] == True].groupby('ad_cfid').agg(f).reset_index()
    df_deseg = df[df['mask'] == True].groupby('ad_cfid')['base_seg_id'].apply(list).reset_index(name='de_targeted')
    #return df_origin, df_remain, df_detarg
    KPI_improve = (df_remain['cost_per_conversion'] - df_origin['cost_per_conversion'])/df_origin['cost_per_conversion']
    impression_lost = (df_remain['impressions'] - df_origin['impressions'])/df_origin['impressions']
    seg_lost = (df_remain['base_seg_id'] - df_origin['base_seg_id'])/df_origin['base_seg_id']
    df_origin['new_KPI'] = df_remain['cost_per_conversion'] 
    df_origin['segment_delta'] = seg_lost
    df_origin['KPI_delta'] = KPI_improve 
    df_origin['impressions_delta'] = impression_lost
    df_origin['method'] = method_name
    #return df_origin
    return df_origin, df_remain, df_detarg,df_deseg

    


# In[6]:


dfd = metricdata.groupby('ad_cfid')['targ_method'].value_counts().unstack().fillna(0).reset_index()


# In[7]:


dfd 


# In[8]:


#fd = dfd[(dfd.behavioral > 0)]
#adid= fd.ad_cfid
#newdata = metricdata[metricdata.ad_cfid.isin (adid)] 
newdata = metricdata
newdata.ad_cfid.nunique()


# In[9]:


f_endal = {'base_seg_id': 'count','cost_per_conversion':     ['mean','std', lambda x: x.mean() + 2* x.std()], 'impressions': 'sum'}


# In[10]:


f_noble = f_noble = {'base_seg_id':            ['count', lambda x: abs(norm.ppf(1/(4*x.count())))],            'cost_per_conversion':['mean','std'], 'impressions': 'sum'}


# In[11]:


mid_endal = RuleFun(newdata, f_endal)
mid_endal.shape


# In[12]:


mid_noble = RuleFun(newdata[newdata.targ_method != 'untargeted'], f_noble)


# In[13]:


mid_noble.shape


# In[14]:


result_endal = DetargFun(mid_endal, 'cost_per_conversion', 'endal')[0]


# In[15]:


result_endal


# In[16]:


detarget_endal = DetargFun(mid_endal, 'cost_per_conversion', 'endal')[3]


# In[17]:


detarget_endal.head(2)


# In[18]:


def detarget_str(value):
    
    return ("AND NOT (" + ' '.join("s={} OR".format(n) for n in value ) + ')')


# In[19]:


detarget_endal['detarget_string'] = detarget_endal['de_targeted'].apply(detarget_str)


# In[20]:


#detarget_endal['count'] = detarget_endal['de_targeted']


# In[21]:


detarget_endal['count'] = detarget_endal['de_targeted'].str.len()


# In[22]:


detarget_endal.to_csv('midflight_output/HBO_28days_detarget.csv', index = False)


# In[30]:


"AND NOT (" + ' '.join("s={} OR".format(n) for n in detarget_endal.de_targeted[5] ) + ')'


# In[24]:


result_endal.to_csv('midflight_output/HBO_28days.csv')


# In[25]:


result_endal


# In[145]:


result_noble = DetargFun(mid_noble, 'deviation', 'noble')


# In[146]:


result_noble.shape


# In[147]:


result_list = [result_endal, result_noble]

df_comp = pd.concat(result_list)


# In[148]:


df_comp.head(2)


# In[149]:


df_summary = df_comp.groupby(['method'])['segment_delta', 'KPI_delta', 'impressions_delta'].mean()


# In[150]:


df_summary


# In[ ]:




