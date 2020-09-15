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
redshift_endpoint1 = "device-advertising.cqporcaaztp8.us-east-1.redshift.amazonaws.com"
redshift_user1 = "zhenlang"
redshift_pass1 = "PizzaTaco1"
port1 = 8192
dbname1 = "deviceadvertising"
from sqlalchemy import create_engine
from sqlalchemy import text
engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" % (redshift_user1, redshift_pass1, redshift_endpoint1, port1, dbname1)
engine1 = create_engine(engine_string)


# In[417]:



endal = pd.read_sql_query('''

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
--and st.advertiser_name IN ('HBO - Television - US', 'Cinemax - US')
AND st.advertiser_name = 'HBO - Television - US'
AND st.marketplace_id = 1
--AND st.start_dt_utc > to_date('20190731','YYYYMMDD')
--AND st.start_dt_utc < to_date('20190811','YYYYMMDD')
AND st.end_dt_utc < to_date('20190930','YYYYMMDD') 
--AND DATEDIFF(day, st.start_dt_utc, st.dim_date) > 14
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

SELECT * 
FROM ads_data
WHERE impressions >= 100
ORDER BY dim_date ASC;

''', engine1)


# In[418]:


endal.shape


# In[401]:



endal_test = pd.read_sql_query('''

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
--and st.advertiser_name IN ('HBO - Television - US', 'Cinemax - US')
AND st.advertiser_name = 'HBO - Television - US'
AND st.marketplace_id = 1
AND st.start_dt_utc > to_date('20190631','YYYYMMDD')
AND st.start_dt_utc < to_date('20190704','YYYYMMDD')
AND st.end_dt_utc < to_date('20190930','YYYYMMDD') 
--AND DATEDIFF(day, st.start_dt_utc, st.dim_date) > 14
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

SELECT * 
FROM ads_data
WHERE impressions >= 100
ORDER BY dim_date ASC;

''', engine1)


# In[402]:


endal_test.apply(pd.Series.nunique)


# In[403]:


endal_test.shape


# In[419]:


endal.apply(pd.Series.nunique)


# In[404]:


endal_10 = endal_test.copy()
endal_10['conversions'].fillna(0, inplace = True)
endal_10['cr'] = endal_10['conversions']/endal_10['impressions']

print(endal_10['cr'].mean(), endal_10['cr'].std())


# In[405]:


# split data by campaign id
gb = endal_10.groupby('campaign_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
#shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*1)
#camplist_test = shuffle[:camp_test_size].tolist()


# In[406]:


len(df_camp)


# In[244]:


camp = endal_10.groupby('campaign_cfid')['base_seg_id'].nunique().reset_index()


# In[245]:


camp.base_seg_id


# In[210]:


#camplist_test


# In[211]:


camp_test_list = [df_camp[i] for i in camplist_test]


# In[198]:


camp_test_list[1]


# In[309]:


endal.apply(pd.Series.nunique)


# In[310]:


endal.shape


# In[311]:


adid = endal.ad_cfid.unique()
adid


# In[86]:





# In[126]:


start = endal.start_dt_utc.unique()
start


# In[74]:





# In[414]:


cpid = endal.campaign_cfid.unique()


# In[415]:


cpid


# In[79]:


#ccc = endal_1.groupby(['campaign_cfid','base_seg_id'])['cr'].sum().reset_index()


# In[416]:


endal_1 = endal.copy()
endal_1['conversions'].fillna(0, inplace = True)
endal_1['cr'] = endal_1['conversions']/endal_1['impressions']

print(endal_1['cr'].mean(), endal_1['cr'].std())


# In[305]:


# ccc = endal_1.groupby(['campaign_cfid','base_seg_id'])['cr'].sum().reset_index()
# ce = ccc[ccc.cr == 0].base_seg_id


# In[313]:


# c = Counter(ce)
# k = {k:v for k, v in c.items() if v >= int(20 * 0.4)}


# In[314]:


df_summary = endal_1.groupby('base_seg_id').aggregate({'impressions':np.mean, 'cr': np.mean}).reset_index()


# In[315]:


df_summary.head(2)


# In[316]:


df_summary.to_csv('results/hbo_seg_info.csv', index= False)


# In[317]:


## segments that don't conversions across all records. So that they can be removed
segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.mean()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))
# Remove these segments from the data
seg_mask = endal_1['base_seg_id'].isin (nonsegid)
endal_seg = endal_1[~seg_mask]


# In[318]:


# split data by campaign id
gb = endal_1.groupby('campaign_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*1)
camplist_test = shuffle[:camp_test_size].tolist()
camplist_train = shuffle[camp_test_size:].tolist()


# In[319]:


camp_train_list = [df_camp[i] for i in camplist_train]
camp_test_list = [df_camp[i] for i in camplist_test]
#endal_train = pd.concat(camp_train_list)
#endal_test = pd.concat(camp_test_list)


# In[320]:


#camp_test_list[1]


# In[321]:


len(camplist_test)


# In[ ]:





# In[322]:


endal_comp = endal_seg[['base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape


# In[324]:


# onehot encoded seg_id
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix='seg')


# In[325]:


endal_encoded.head(3)


# In[326]:


endal_encoded.shape


# In[327]:


from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[ ]:





# In[328]:


from functools import reduce
from collections import Counter

def BottomList (dataframe, colname, pct):
    groupby_df = dataframe.groupby(colname, as_index=False)['cr'].agg(['mean', 'count']).sort_values('mean',ignore_index=False).reset_index()
    smalln = math.ceil(len(groupby_df)*pct)
    seg_removed_bottom = groupby_df.nsmallest(smalln,'mean')[colname].to_list()
    bottom_segid = [i for i in seg_removed_bottom]
    return bottom_segid

def CoefRank (df, ranka):
    names = list(df.columns[ranka].values)
    result = list(map(lambda x: re.findall('\d+', x), names))
    flattened = [val for sublist in result for val in sublist]
    return(flattened)

def CoefNeg (df, coef):
    index = [i for i, e in enumerate(coef) if e < 0]
    names = list(df.columns[index].values)
    result = list(map(lambda x: re.findall('\d+', x), names))
    flattened = [val for sublist in result for val in sublist]
    return(flattened)

def cvResult(avglist, diclist, cvn, f):
    index_max = np.argmax(avglist)
    seg = diclist[index_max]
    avgDict = {}
    for k,v in seg.items():
        avgDict[k] = sum(v)/ float(len(v))
    allkeys = [i for s in [d.keys() for d in diclist] for i in s]
    c=Counter(allkeys)
    k = {k:v for k, v in c.items() if v >= int(cvn * f)}
    segff = [*k]
    dict_seg_r = {k: v for k, v in avgDict.items() if k in segff}
    return (dict_seg_r)

def Intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def improve_f(df, remove_id):
    test_data_remove = df[~df['base_seg_id'].isin(remove_id)]
    improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return improve * 100

def marg_improve (df, remove_id):
    is_removed = df['base_seg_id']== remove_id
    df_new = df[~is_removed]
    imp = (df_new['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return imp


# In[329]:


seg_uniques = df_summary.base_seg_id
len(seg_uniques)


# In[330]:


marg_imp = [marg_improve (df_summary, i) for i in df_summary.base_seg_id]
df_summary['marg_imp'] = marg_imp
df_summary.shape


# In[331]:


df_summary[df_summary['marg_imp'] > 0].shape


# In[332]:


df_summary.to_csv('results/hbo_seg_info.csv', index= False)


# In[314]:



# Benchmark: select the bottom 10% of the train data as the prediction + Cross-validation
start_time = time.monotonic()
n = 5
seg_iteration = []
overlap_rate_mean = []
imp_mean = []
ii = 1
random.seed(30)
for i in range (n):
    seg_store = []
    overlap_rate = []
    imp_rate= []
    for train_indice, test_indice in sss.split(endal_seg):
        df_train = endal_seg.iloc[train_indice]
        df_test = endal_seg.iloc[test_indice]
        bottom_seg = BottomList(df_train, 'base_seg_id', 0.1)
        test_boid = BottomList(df_test, 'base_seg_id', 0.1)
        intersect = len(Intersection(test_boid, bottom_seg))/len(test_boid)
        imp = improve_f(df_test, bottom_seg)
        overlap_rate.append(intersect)
        seg_store.append(bottom_seg)
        imp_rate.append(imp)
    single_list = reduce(lambda x,y: x+y, seg_store)
    c= Counter(single_list)
    k = {k:v for k, v in c.items() if v >= int(0.7*i)}
    segff = [*k]
    seg_iteration.append(segff)
    overlap_rate_mean.append(np.mean(overlap_rate))
    imp_mean.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    Test_seg_test = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
    index_max_bench = np.argmax(overlap_rate_mean)
    seg_bench = seg_iteration[index_max_bench]
    intersect = len(Intersection(seg_bench, Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[69]:


# Cross validation for linear regression
start_time = time.monotonic()
n = 5

overlap_rate_mean_lin = []
imp_mean_lin = []
coefdict_lin = []
ii = 1
alpha = 0.000000001
random.seed(30)

for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(endal_seg):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        list_coef = lin_reg.coef_
        test_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.1)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(endal_seg.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_lin.append(dd)
    overlap_rate_mean_lin.append(np.mean(overlap_rate))
    imp_mean_lin.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_lin =(cvResult(overlap_rate_mean_lin, coefdict_lin, i, 0.7 ))
    Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_lin.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t, intersect)
    ii +=1


# In[70]:


segr_lin5 = cvResult(overlap_rate_mean_lin, coefdict_lin, 5, 0.7)


# In[208]:


seg_suggest = [*segr_lin15]
len(seg_suggest)


# In[209]:


test_lsn = len(camp_test_list)
overlap_rep = []
est_seg_lin = seg_suggest
for i in range (test_lsn):
    df_test = camp_test_list[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>=1000:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
        ovlr = len(Intersection(est_seg_lin, Test_seg_test))/len(Test_seg_test)
        overlap_rep.append(ovlr)


# In[203]:





# In[210]:


np.mean(overlap_rep)


# In[74]:


print('mean of overlap rate: {0: .2f} the std: {1: .2f}'.format(np.mean(overlap_rep), np.std(overlap_rep)))


# In[75]:


(pd.DataFrame.from_dict(data=segr_lin5, orient='index')   .to_csv('segr_lin5_HBO.csv', header=False))


# In[333]:



# Cross validation for Ridge regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_ridge = []
imp_mean_ridge = []
coefdict_ridge = []
ii = 1
alpha = 1
random.seed(30)
for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(endal_seg):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        list_coef = ridge.coef_
        test_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.05)
        train_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(endal_seg.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_ridge.append(dd)
    overlap_rate_mean_ridge.append(np.mean(overlap_rate))
    imp_mean_ridge.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.9 ))
    Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
    intersect = len(Intersection(segr_ridge.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t, intersect)
    ii +=1


# In[334]:


segr_ridge5 = cvResult(overlap_rate_mean_ridge, coefdict_ridge, 15, 0.9)


# In[420]:


len(segr_ridge5)


# In[394]:


[*segr_ridge5][0:35]


# In[395]:


suggestseg = [*segr_ridge5][0:34] + nonsegid


# In[396]:


len(suggestseg)


# In[181]:


#coefdict_ridge


# In[337]:



Test_seg = BottomList(endal_seg, 'base_seg_id', 0.05)
len(Intersection(segr_ridge5.keys(), Test_seg))/len(Test_seg)


# In[338]:


len(Test_seg)


# In[339]:


sedf= pd.DataFrame(list(segr_ridge5.items()),
                      columns=['Seg_id','Coef_value'])


# In[ ]:





# In[340]:


names_id = list(map(lambda x: re.findall('\d+', x), df_summary.base_seg_id))
flattened = [val for sublist in names_id for val in sublist]
df_summary['names_id'] = flattened


# In[341]:


hbo_seg_df = pd.merge(sedf, 
         df_summary, 
         how='left', left_on='Seg_id', right_on = 'names_id')


# In[383]:


hbo_seg_df_f = hbo_seg_df[(hbo_seg_df['marg_imp']>0)]


# In[384]:


hbo_seg_df_f.shape


# In[344]:


hbo_seg_df_f.to_csv('results/candidatehboall.csv', index = False)


# In[385]:


hbo_seg_df_f.to_csv('results/segr5_hboall.csv', index= False)


# In[386]:


seg_suggest = [*segr_ridge5]
len(seg_suggest)


# In[387]:


seg_bot= BottomList(endal_1,'base_seg_id', 0.05 )
len(seg_bot)


# In[407]:


test_lsn = len(df_camp)


# In[408]:


test_lsn


# In[409]:


test_lsn = len(df_camp)
overlap_rep_ridge = []
se = []
imp = []
est_seg_ridge = suggestseg
for i in range (test_lsn):
    df_test = df_camp[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>100:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.05)
        ovlr = len(Intersection(est_seg_ridge, Test_seg_test))/len(Test_seg_test)
        se.append(segn)
        overlap_rep_ridge.append(ovlr)
        if df_test['cr'].mean()!=0:
            impr = improve_f(df_test,est_seg_ridge)
            #impr = improve_f(df_test,Test_seg_test)
            imp.append(impr)
    


# In[411]:


np.mean(imp)


# In[413]:


np.std(imp)


# In[354]:


len(overlap_rep_ridge)


# In[355]:


np.mean(overlap_rep_ridge)


# In[356]:


np.std(overlap_rep_ridge)


# In[ ]:


len(Intersection(est_seg_ridge, est_seg_lin))


# In[66]:


len(est_seg_ridge)


# In[67]:


print('mean of overlap rate: {0: .2f} the std: {1: .2f}'.format(np.mean(overlap_rep), np.std(overlap_rep)))


# In[68]:


(pd.DataFrame.from_dict(data=segr_ridge15, orient='index')   .to_csv('segr_ridge15_HBO.csv', header=False))


# In[17]:


endal_encoded.head(2)


# In[18]:





# In[19]:


for train_indice, test_indice in ss.split(Train):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(Train.iloc[train_indice]['base_seg_id'].unique())


# In[62]:


train_indice


# In[64]:


pd.DataFrame(train_indice).to_csv("hbo_trainindice.csv")


# In[65]:


pd.DataFrame(test_indice).to_csv("hbo_testindice.csv")


# In[20]:


Test_seg = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
len(Test_seg)


# In[21]:


Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Test_seg_test)


# In[22]:


# Linear regression 
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_coef = lin_reg.coef_
y_pred = lin_reg.fit(X_train, y_train).predict(X_test) 
train_predictions = lin_reg.predict(X_train)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
a_lin = lin_coef.argsort()
bo10_segn = int(seg_n*0.1)
seg_lin = CoefRank(X_train, a_lin[0:bo10_segn+1])


# In[23]:


print(lin_rmse, len(Intersection(seg_lin, Test_seg_test))/len(Test_seg_test))


# In[ ]:


# Import necessary modules
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


alpha_space = np.logspace(-8, 0, 10)
reg_CV = RidgeCV(alphas=alpha_space, cv=5)
reg_CV.fit(X_train, y_train)


# In[ ]:


reg_CV.alpha_
reg_CV.coef_


# In[ ]:



# Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ridge = Ridge(alpha= 1)
ridge_y_pred = ridge.fit(X_train, y_train).predict(X_test) 
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_rmse
ridge_coef = ridge.coef_
ridge_coef
a_ridge = ridge_coef.argsort()
seg_ridge = CoefRank(X_train, a_ridge[0:bo10_segn+1])


# In[ ]:


print(ridge_rmse, len(Intersection(seg_ridge, Test_seg_test))/len(Test_seg_test))


# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

alpha_space = np.logspace(-8, 0, 5)
lasso_CV = LassoCV(alphas=alpha_space, cv=5)
lasso_CV.fit(X_train, y_train)


# In[ ]:


lasso_CV.alpha_


# In[30]:


# Lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

lasso = Lasso(alpha=0.000000001)
lasso_y_pred = lasso.fit(X_train, y_train).predict(X_test) 
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_rmse
lasso_coef = lasso.coef_
lasso_coef
a_lasso = lasso_coef.argsort()
seg_lasso = CoefRank(X_train, a_lasso[0:bo10_segn+1])


# In[31]:


print(lasso_rmse, len(Intersection(seg_lasso, Test_seg_test))/len(Test_seg_test))


# In[ ]:


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

l1_space = np.linspace(0.1,1,10)
alpha_space = np.logspace(-8, 0, 5)

param_grid = {'l1_ratio': l1_space,
             'alpha':alpha_space}

elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)


# In[34]:


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
alpha = 0.000000001
enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
enet_y_pred = enet.fit(X_train, y_train).predict(X_test)
enet_mse = mean_squared_error(y_test, enet_y_pred)
enet_rmse = np.sqrt(enet_mse)
enet_rmse
enet_coef  = enet.coef_
a_enet = enet_coef.argsort()
seg_enet = CoefRank(X_train, a_enet[0:bo10_segn+1])


# In[33]:


print(enet_rmse, len(Intersection(seg_enet, Test_seg_test))/len(Test_seg_test))


# In[35]:


# Benchmark: select the bottom 10% of the train data
Train_seg = BottomList(Train, 'base_seg_id', 0.1)
len(Intersection(Train_seg, Test_seg_test))/len(Test_seg_test)


# In[40]:


import time
from datetime import timedelta


# In[41]:


sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[42]:


# Benchmark: select the bottom 10% of the train data as the prediction + Cross-validation
start_time = time.monotonic()
n = 15
seg_iteration = []
overlap_rate_mean = []
imp_mean = []
ii = 1
random.seed(30)
for i in range (n):
    seg_store = []
    overlap_rate = []
    imp_rate= []
    for train_indice, test_indice in sss.split(Train, Train['campaign_cfid']):
        df_train = Train.iloc[train_indice]
        df_test = Train.iloc[test_indice]
        bottom_seg = BottomList(df_train, 'base_seg_id', 0.1)
        test_boid = BottomList(df_test, 'base_seg_id', 0.1)
        intersect = len(Intersection(test_boid, bottom_seg))/len(test_boid)
        imp = improve_f(df_test, bottom_seg)
        overlap_rate.append(intersect)
        seg_store.append(bottom_seg)
        imp_rate.append(imp)
    single_list = reduce(lambda x,y: x+y, seg_store)
    c= Counter(single_list)
    k = {k:v for k, v in c.items() if v >= int(0.7*5)}
    segff = [*k]
    seg_iteration.append(segff)
    overlap_rate_mean.append(np.mean(overlap_rate))
    imp_mean.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    index_max_bench = np.argmax(overlap_rate_mean)
    seg_bench = seg_iteration[index_max_bench]
    intersect = len(Intersection(seg_bench, Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[43]:


index_max_bench = np.argmax(overlap_rate_mean)
seg_bench = seg_iteration[index_max_bench]
len(Intersection(Test_seg_test,seg_bench))/len(Test_seg_test)


# In[55]:


# Cross validation for linear regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_lin = []
imp_mean_lin = []
coefdict_lin = []
ii = 1
alpha = 0.000000001
random.seed(30)

for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(Train, Train['campaign_cfid']):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        #enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
        #enet.fit(X_train, y_train)
        #list_coef = enet.coef_
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        list_coef = lin_reg.coef_
        #lasso = Lasso(alpha=alpha)
        #lasso.fit(X_train, y_train)
        #list_coef = lasso.coef_
        test_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(Train.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_lin.append(dd)
    overlap_rate_mean_lin.append(np.mean(overlap_rate))
    imp_mean_lin.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_lin =(cvResult(overlap_rate_mean_lin, coefdict_lin, i, 0.7 ))
    #Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_lin.keys(), Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[50]:


segr_lin15 = cvResult(overlap_rate_mean_lin, coefdict_lin, 10, 0.7)
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_lin10.keys(), Test_seg_test))/len(Test_seg_test)


# In[51]:


print(len(Intersection(segr_lin.keys(), Test_seg_test))/len(Test_seg_test))
print(len(Intersection(segr_lin10.keys(), Test_seg_test))/len(Test_seg_test))
#print(len(Intersection(segr_lin15.keys(), Test_seg_test))/len(Test_seg_test))


# In[30]:


(pd.DataFrame.from_dict(data=segr_lin_15, orient='index')   .to_csv('segr_lin15_HBO.csv', header=False))


# In[ ]:



# Cross validation for Lasso regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_lasso = []
imp_mean_lasso = []
coefdict_lasso = []
ii = 1
alpha = 0.000000001
random.seed(30)
for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(Train, ):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        #enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
        #enet.fit(X_train, y_train)
        #list_coef = enet.coef_
        #lin_reg = LinearRegression()
        #lin_reg.fit(X_train, y_train)
        #list_coef = lin_reg.coef_
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        list_coef = lasso.coef_
        test_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(Train.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_lasso.append(dd)
    overlap_rate_mean_lasso.append(np.mean(overlap_rate))
    imp_mean_lasso.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_lasso =(cvResult(overlap_rate_mean_lasso, coefdict_lasso, i, 0.7 ))
    #Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_lasso.keys(), Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[45]:


segr_lasso10 = cvResult(overlap_rate_mean_lasso, coefdict_lasso, 10, 0.7 )
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_lasso_10.keys(), Test_seg_test))/len(Test_seg_test)


# In[ ]:


print(len(Intersection(segr_lasso.keys(), Test_seg_test))/len(Test_seg_test))
print(len(Intersection(segr_lasso10.keys(), Test_seg_test))/len(Test_seg_test))
print(len(Intersection(segr_lasso15.keys(), Test_seg_test))/len(Test_seg_test))


# In[47]:


(pd.DataFrame.from_dict(data=segr_lin_10, orient='index')   .to_csv('segr_lasso10_HBO.csv', header=False))


# In[59]:


from sklearn.linear_model import Ridge


# In[60]:



# Cross validation for Ridge regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_ridge = []
imp_mean_ridge = []
coefdict_ridge = []
ii = 1
alpha = 1
random.seed(30)
for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(Train, Train['campaign_cfid']):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        list_coef = ridge.coef_
        test_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(Train.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_ridge.append(dd)
    overlap_rate_mean_ridge.append(np.mean(overlap_rate))
    imp_mean_ridge.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.7 ))
    #Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_ridge.keys(), Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[ ]:





# In[57]:



# Cross validation for elastic net regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_enet = []
imp_mean_enet = []
coefdict_enet = []
ii = 1
alpha = 0.000000001
random.seed(30)
for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    for train_indice, test_indice in sss.split(Train, Train['campaign_cfid']):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
        enet.fit(X_train, y_train)
        list_coef = enet.coef_
        #lin_reg = LinearRegression()
        #lin_reg.fit(X_train, y_train)
        #list_coef = lin_reg.coef_
        #lasso = Lasso(alpha=alpha)
        #lasso.fit(X_train, y_train)
        #list_coef = lasso.coef_
        test_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
        a = list_coef.argsort()
        coefcollect = list_coef[a[0:len(train_boid)+1]]
        ml_segid = CoefRank(X_train, a[0:len(train_boid)+1])
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(Train.iloc[test_indice], ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_enet.append(dd)
    overlap_rate_mean_enet.append(np.mean(overlap_rate))
    imp_mean_enet.append(np.mean(imp_rate))
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    segr_enet =(cvResult(overlap_rate_mean_enet, coefdict_enet, i, 0.7 ))
    #Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_enet.keys(), Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[41]:


segr_enet10 = cvResult(overlap_rate_mean_enet, coefdict_enet, 6, 0.7 )
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_enet10.keys(), Test_seg_test))/len(Test_seg_test)


# In[ ]:


print(len(Intersection(segr_enet.keys(), Test_seg_test))/len(Test_seg_test))
print(len(Intersection(segr_enet10.keys(), Test_seg_test))/len(Test_seg_test))
print(len(Intersection(segr_enet15.keys(), Test_seg_test))/len(Test_seg_test))


# In[237]:


# test_lsn = len(camp_test_list)
# overlap_rep = []
# est_seg = segr.keys()
# for i in range (test_lsn):
#     df_test = camp_test_list[i]
#     Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
#     ovlr = len(Intersection(est_seg, Test_seg_test))/len(Test_seg_test)
#     overlap_rep.append(ovlr)


# In[43]:


#print(np.mean(overlap_rep), np.std(overlap_rep))


# In[44]:


#overlap_rep


# In[ ]:


################################### Tree-based models #################################


# In[40]:


#Select one campagin's data 
Train, Test = train_test_split (df_camp[13], test_size= 0.2, random_state = 42)
Train.shape


# In[41]:


endal_comp = Train[['base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix='seg')


# In[44]:


ss = ShuffleSplit(n_splits=1, test_size= 0.2)
for train_indice, test_indice in ss.split(Train):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(Train.iloc[train_indice]['base_seg_id'].unique())


# In[43]:


Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Test_seg_test)


# In[49]:


X_test.head(2)
#y_train.head(2)


# In[45]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth = 4,
                                min_samples_leaf = 0.1,
                                random_state = 3)


# In[46]:


tree_y_pred = tree_reg.fit(X_train, y_train).predict(X_test)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_y_pred))
tree_rmse


# In[47]:


print(tree_reg.score(X_train, y_train))
print(tree_reg.score(X_test, y_test))


# In[48]:


importances_dr = pd.Series(tree_reg.feature_importances_, index = X_train.columns)
sorted_importance_dr = importances_dr.sort_values(ascending = False)


# In[49]:


sorted_importance_dr


# In[50]:


from sklearn.ensemble import RandomForestRegressor 

# Instantiate rf
rf = RandomForestRegressor(n_estimators=4,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 


# In[51]:


rf_y_pred = rf.predict (X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_rmse


# In[52]:


importances_dr = pd.Series(rf.feature_importances_, index = X_train.columns)
sorted_importance_dr = importances_dr.sort_values(ascending = False)
sorted_importance_dr


# In[53]:


nseg= int(len(X_train.columns)*0.1)


# In[54]:


def featureRank (featuresrank,dfindex, nseg):
    importances_dr = pd.Series(featuresrank, index = dfindex)
    sorted_importance_dr = importances_dr.sort_values(ascending = True)
    ids = sorted_importance_dr[0:nseg+1].index.tolist()
    values = sorted_importance_dr[0:nseg+1].values.tolist()
    result = list(map(lambda x: re.findall('\d+', x), ids))
    flattened = [val for sublist in result for val in sublist]
    seg_dic = dict(zip(flattened, values))
    return seg_dic
    


# In[117]:


# id = sorted_importance_dr[0:nseg+1].index.tolist()
# values = sorted_importance_dr[0:nseg+1].values.tolist()
# result = list(map(lambda x: re.findall('\d+', x), id))
# flattened = [val for sublist in result for val in sublist]
# rf_seg = flattened


# In[55]:


seg_rf = featureRank(rf.feature_importances_, X_train.columns, nseg)


# In[56]:


len(Intersection(Test_seg_test,seg_rf))/len(Test_seg_test)


# In[57]:


# Gradient boosting
from sklearn.ensemble import GradientBoostingRegressor


# In[58]:


gbt = GradientBoostingRegressor(n_estimators = 40, max_depth = 4, random_state = 2)


# In[59]:


gbt.fit(X_train, y_train)
gbt_y_pred = gbt.predict(X_test)
gbt_rmse = np.sqrt(mean_squared_error(y_test, gbt_y_pred))


# In[60]:


gbt_rmse


# In[61]:


seg_gbt = featureRank(gbt.feature_importances_, X_train.columns, nseg)


# In[62]:


gbt_seg = seg_gbt.keys()


# In[63]:


len(Intersection(Test_seg_test,gbt_seg))/len(Test_seg_test)


# In[64]:


# Stochastic gradient boosting
sgbt = GradientBoostingRegressor (max_depth = 1, 
                                  subsample = 0.8,
                                 max_features = 0.2,
                                 n_estimators = 100,
                                 random_state = 32)


# In[65]:


sgbt.fit(X_train, y_train)


# In[66]:


sgbt_y_pred = sgbt.predict(X_test)
sgbt_rmse = np.sqrt(mean_squared_error(y_test, sgbt_y_pred))
sgbt_rmse


# In[67]:


seg_sgbt = featureRank(sgbt.feature_importances_, X_train.columns, nseg)


# In[68]:


len(Intersection(Test_seg_test,seg_sgbt.keys()))/len(Test_seg_test)


# In[ ]:





# In[84]:


(pd.DataFrame.from_dict(data=segr_enet, orient='index')
   .to_csv('segr_lasso6_HBO.csv', header=False))


# In[81]:


Intersection(segr_enet, segr_lin)


# In[83]:


R="\n".join(nonsegid)
f = open('HBO_noconversion.csv','w')
f.write(R)
f.close()


# In[ ]:


# 

