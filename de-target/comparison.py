#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[42]:



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
VAP_AGG.SEGMENT_PVC_TRAFFIC st
LEFT JOIN VAP_AGG.SEGMENT_PVC_CONVERSION sc
ON st.base_seg_id = sc.base_seg_id
and st.dim_date= sc.dim_Date
and st.marketplace_id=sc.marketplace_id
and st.ad_cfid = sc.ad_cfid
and st.targ_method = sc.targ_method
and st.seg_name = sc.seg_name
--where st.targ_method != 'untargeted'  -- Filter on only untargeted segments
--AND st.advertiser_name IN ('Cinemax - US', 'HBO - Television - US')
where
st.start_dt_utc > to_date('20190101','YYYYMMDD')
--AND st.start_dt_utc < to_date('20190815','YYYYMMDD')
AND st.end_dt_utc < to_date('20191230','YYYYMMDD')
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
sum(st.impressions) as impressions,
sum(clicks) as clicks,
sum(conversions) as conversions,
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion
--nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ELIGIBLE_CAMPAIGNS_PVC a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and conversions > 0
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
property_type''', engine1)


# In[43]:


metricdata.shape


# In[44]:


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
    #return df_origin, df_remain, df_detarg
    KPI_improve = (df_remain['cost_per_conversion'] - df_origin['cost_per_conversion'])/df_origin['cost_per_conversion']
    impression_lost = (df_remain['impressions'] - df_origin['impressions'])/df_origin['impressions']
    seg_lost = (df_remain['base_seg_id'] - df_origin['base_seg_id'])/df_origin['base_seg_id']
    df_origin['new_KPI'] = df_remain['cost_per_conversion'] 
    df_origin['segment_delta'] = seg_lost
    df_origin['KPI_delta'] = KPI_improve 
    df_origin['impressions_delta'] = impression_lost
    df_origin['method'] = method_name
    df_origin = df_origin.rename(columns={"base_seg_id": "seg_count"})
    return df_origin
    #return df_origin, df_remain, df_detarg

    


# In[45]:


dfd = metricdata.groupby('ad_cfid')['targ_method'].value_counts().unstack().fillna(0).reset_index()


# In[67]:


np.mean((fd['behavioral'] - fd['untargeted'])/fd['behavioral'])


# In[46]:


metricdata.base_seg_id.nunique()


# In[47]:


metricdata.ad_cfid.nunique()


# In[48]:


metricdata.groupby(['targ_method'])['base_seg_id'].nunique()


# In[49]:


4358 + 757 + 12


# In[50]:


fd = dfd[(dfd.behavioral > 0)]
adid= fd.ad_cfid
newdata = metricdata[metricdata.ad_cfid.isin (adid)] 
newdata.ad_cfid.nunique()


# In[51]:


newdata.head(2)


# In[52]:


f_endal = {'base_seg_id': 'count','cost_per_conversion':     ['mean','std', lambda x: x.mean() + 2* x.std()], 'impressions': 'sum'}


# In[53]:


f_noble = f_noble = {'base_seg_id':            ['count', lambda x: abs(norm.ppf(1/(4*x.count())))],            'cost_per_conversion':['mean','std'], 'impressions': 'sum'}


# In[54]:


mid_endal = RuleFun(newdata[newdata.targ_method == 'untargeted'], f_endal)
mid_endal.shape


# In[55]:


mid_noble = RuleFun(newdata[newdata.targ_method != 'untargeted'], f_noble)


# In[56]:


mid_noble.shape


# In[ ]:





# In[57]:


result_endal = DetargFun(mid_endal, 'cost_per_conversion', 'endal')


# In[58]:


result_endal.head(3)


# In[59]:


result_noble = DetargFun(mid_noble, 'deviation', 'noble')


# In[60]:


result_noble.shape


# In[61]:


result_list = [result_endal, result_noble]

df_comp = pd.concat(result_list)


# In[63]:


df_comp.shape


# In[64]:


df_comp.groupby(['ad_cfid', 'method'])['impressions'].mean()


# In[65]:


df_summary = df_comp.groupby(['method'])['segment_delta', 'KPI_delta', 'impressions_delta'].agg(['mean', 'std'])


# In[66]:


df_summary


# In[208]:


endal_cmax = pd.read_sql_query('''

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
VAP_AGG.SEGMENT_PVC_TRAFFIC st
LEFT JOIN VAP_AGG.SEGMENT_PVC_CONVERSION sc
ON st.base_seg_id = sc.base_seg_id
and st.dim_date= sc.dim_Date
and st.marketplace_id=sc.marketplace_id
and st.ad_cfid = sc.ad_cfid
and st.targ_method = sc.targ_method
and st.seg_name = sc.seg_name
where st.targ_method = 'untargeted'  -- Filter on only untargeted segments
AND st.advertiser_name ='Cinemax - US'
AND st.start_dt_utc > to_date('20190630','YYYYMMDD')
--AND st.start_dt_utc < to_date('20190815','YYYYMMDD')
AND st.end_dt_utc < to_date('20190930','YYYYMMDD')
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
sum(st.impressions) as impressions,
sum(clicks) as clicks,
sum(conversions) as conversions,
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion
--nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ELIGIBLE_CAMPAIGNS_PVC a ON st.ad_cfid = a.cfid
    where st.impressions > 100
   -- and conversions >= 0
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
property_type''', engine1)


# In[209]:


endal_cmax.ad_cfid.nunique()


# In[210]:


pre_flight = pd.read_csv('output/seg_cinemax.csv')
nonseg = pd.read_csv('output/nonsegid_cinemax.csv')


# In[263]:


seg =  nonseg.base_seg_id.tolist()


# In[264]:


mask = endal_cmax.base_seg_id.isin (seg)


# In[265]:


endal_cmax.shape


# In[266]:


mask.sum()


# In[268]:


pre_de = endal_cmax[~mask]
pre = pre_de.groupby('ad_cfid')['cost_per_conversion'].mean().reset_index()
pre
orig = endal_cmax.groupby('ad_cfid')['cost_per_conversion'].mean().reset_index()


# In[269]:


se = (orig.cost_per_conversion - pre.cost_per_conversion)
np.mean(se)


# In[270]:


pre_de.shape


# In[271]:


#endal_mid = endal_cmax[endal_cmax.conversions >0]
endal_mid = endal_cmax
endal_mid.shape


# In[236]:


#pre_de_mid = pre_de[pre_de.conversions > 0]
pre_de_mid = pre_de
pre_de_mid.shape


# In[260]:


nonpred_endal = RuleFun(endal_mid, f_endal)
d1 = nonpred_endal.groupby('ad_cfid')['cost_per_conversion'].mean().reset_index()


# In[262]:


pred_endal = RuleFun(pre_de_mid, f_endal)
d2 = pred_endal.groupby('ad_cfid')['cost_per_conversion'].mean().reset_index()
s = (d1.cost_per_conversion - d2.cost_per_conversion)
np.mean(s)


# In[239]:


result_nonpre_endal = DetargFun(nonpred_endal, 'cost_per_conversion', 'endal_nonpre')
result_pre_endal = DetargFun(pred_endal, 'cost_per_conversion', 'endal_pre')


# In[240]:


result_list_endal = [result_nonpre_endal, result_pre_endal]

df_comp_endal = pd.concat(result_list_endal)


# In[241]:


df_summary_endal = df_comp_endal.groupby(['method'])['segment_delta', 'KPI_delta', 'impressions_delta'].agg(['mean', 'std'])


# In[242]:


df_summary_endal


# In[19]:


endal_mid_flight = pd.read_sql_query('''

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
and st.advertiser_name = 'Cinemax - US'
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
),
metric_ads_conversions as (
select a.*,
avg(cost_per_conversion) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as avg_cpa,
stddev(cost_per_conversion) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as stddev_cpa
from (
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
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion,
nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and st.advertiser_name = 'Cinemax - US'
    and conversions > 0
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
targeting_combined
) a
),
metric_ads_clicks as (
select a.*,
avg(ctr) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as avg_cpa,
stddev(ctr) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as stddev_cpa
from (
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
case when sum(conversions) > 0 then nvl(((sum(st.impressions)/1000)*(max(a.revenue)/(max(a.total_impressions)/1000)))/sum(conversions),0) else 0 end as cost_per_conversion,
nvl((sum(st.clicks)*1000000)/sum(impressions),0) as ctr
from 
ads_data st
JOIN VAP_AGG.ENDAL_PVC_ELIGIBLE_CAMPAIGNS a ON st.ad_cfid = a.cfid
    where st.impressions > 100
    and clicks > 0
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
targeting_combined
) a
)
,filter_metrics_data as ( --- Only consider segments 2 stddev away from mean
select a.*, last_value(type) over (partition by ad_cfid rows between unbounded preceding and unbounded following) as first_type
from (
select cc.*, 'Conv' as type, case when cost_per_conversion >= avg_cpa+(2*stddev_cpa) then 1 else 0 end as tbr from metric_ads_conversions cc
where avg_cpa > 0
--UNION ALL
--select c.*, 'Clicks' as typec,case when ctr < avg_cpa-(stddev_cpa) then 1 else 0 end as tbr from metric_ads_clicks c
--where avg_cpa > 0
) a
)

select entity_id as entityid
, null as customerid
, campaign_cfid as ordercfid
, ad_cfid as lineitemcfid
, 'SEGMENT_REMOVAL' as optimizationlever
, 'REMOVE_UNTARGETED_SEGMENT' as recommendedaction
, '{"TARGETING_STRING":"'|| targeting_Combined || '"}' as curdata
, '{"TARGETING_STRING":"'||listagg('s='||base_seg_id,' OR ') || '"}'  as newdata
,start_dt_utc as eligiblestartdate	
, end_dt_utc as eligibleenddate	
, 'PENDING' as status
, case when property_type = 'Class 1' then 'Class_I' else 'AAP' end as orderstrategy
, case when property_type = 'Class 1' then 'ORDER' else 'LINEITEM' end as updatelevel
, 'NA' as region
, 1 as marketplaceid
, case when property_type = 'Class 1' then 'FORECAST' else 'NO_FORECAST' end as updatestrategy
, 'PVC'as sourcetype
, sysdate as createdAt
, sysdate as updatedAt
from filter_metrics_data
where first_type = type 
and tbr = 1
and base_seg_id not in (32, 20, 21)
--and mod(datediff('day',to_date(start_dt_utc,'YYYYMMDD'),to_date('{RUN_DATE_YYYYMMDD}','YYYMMDD')),7)=1
group by entity_id, campaign_cfid, ad_cfid, targeting_combined, start_Dt_utc, end_dt_utc, property_type
''', engine1)


# In[24]:


endal_mid_flight.curdata[0]


# In[ ]:




