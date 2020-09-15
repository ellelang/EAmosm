#!/usr/bin/env python
# coding: utf-8

# In[149]:


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
from functools import reduce
from collections import Counter
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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


# In[246]:



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
where st.targ_method = 'behavioral'  -- Filter out only untargeted segments
--where st.targ_method = 'untargeted' 
--and st.advertiser_name IN ('HBO - Television - US', 'Cinemax - US')
AND st.advertiser_name = 'Cinemax - US'
AND st.marketplace_id = 1
AND st.start_dt_utc > to_date('20190701','YYYYMMDD')
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

ORDER BY dim_date ASC
LIMIT 3000000;

''', engine1)


# In[247]:


endal.apply(pd.Series.nunique)


# In[248]:


endal.shape


# In[ ]:





# In[234]:


endal_1 = endal.copy()
endal_1['conversions'].fillna(0, inplace = True)
endal_1['cr'] = endal_1['conversions']/endal_1['impressions']

print(endal_1['cr'].mean(), endal_1['cr'].std())


# In[174]:


#ccc = endal_1.groupby(['campaign_cfid','base_seg_id'])['cr'].sum().reset_index()


# In[245]:


segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.sum()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))


# In[177]:


# split data by campaign id
gb = endal_1.groupby('campaign_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
#shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*1)
#camplist_test = shuffle[:camp_test_size].tolist()


# In[178]:


len(df_camp)


# In[167]:





# In[172]:





# In[15]:


df_summary = endal_1.groupby('base_seg_id').aggregate({'impressions':np.mean, 'cr': np.mean}).reset_index()


# In[37]:


df_summary.head(2)
df_summary.shape


# In[36]:





# In[171]:


## segments that don't conversions across all records. So that they can be removed
segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.sum()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))
# Remove these segments from the data
seg_mask = endal_1['base_seg_id'].isin (nonsegid)
endal_seg = endal_1[~seg_mask]


# In[54]:


seg_uniques = df_summary.base_seg_id
len(seg_uniques)


# In[19]:


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

def improve_f(df, remove_index):
    test_data_remove = df[~df['base_seg_id'].isin(remove_index)]
    improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return improve * 100

def marg_improve (df, remove_id):
    is_removed = df['base_seg_id']== remove_id
    df_new = df[~is_removed]
    imp = (df_new['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return imp


# In[56]:


seg_uniques = df_summary.base_seg_id
len(seg_uniques)


# In[75]:


marg_imp = [marg_improve (df_summary, i) for i in df_summary.base_seg_id]
df_summary['marg_imp'] = marg_imp
df_summary.shape


# In[ ]:





# In[60]:


df_summary[df_summary['marg_imp'] > 0]


# In[61]:


df_summary.to_csv('results/cmax_seg_info.csv', index= False)


# In[20]:


# split data by campaign id
gb = endal_1.groupby('campaign_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*1)
camplist_test = shuffle[:camp_test_size].tolist()
camplist_train = shuffle[camp_test_size:].tolist()


# In[21]:


camp_train_list = [df_camp[i] for i in camplist_train]
camp_test_list = [df_camp[i] for i in camplist_test]
#endal_train = pd.concat(camp_train_list)


# In[22]:


len(camp_test_list)


# In[23]:


endal_comp = endal_seg[['base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape


# In[24]:


# onehot encoded seg_id
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix='seg')


# In[25]:


from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[ ]:



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


# In[ ]:


# Cross validation for linear regression
start_time = time.monotonic()
n = 5

overlap_rate_mean_lin = []
imp_mean_lin = []
coefdict_lin = []
ii = 1
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


# In[43]:


segr_lin15 = cvResult(overlap_rate_mean_lin, coefdict_lin, 15, 0.7)


# In[153]:


seg_suggest = [*segr_lin15]+nonsegid
len(seg_suggest)


# In[154]:


test_lsn = len(camp_test_list)
overlap_rep_lin = []
se = []
imp = []
est_seg_lin = seg_suggest
for i in range (test_lsn):
    df_test = camp_test_list[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>=700:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
        ovlr = len(Intersection(est_seg_lin, Test_seg_test))/len(Test_seg_test)
        se.append(segn)
        overlap_rep_lin.append(ovlr)
        if df_test['cr'].mean()!=0:
            impr = improve_f(df_test,est_seg_lin[0:int(segn*0.09)+1])
            #impr = improve_f(df_test,Test_seg_test)
            imp.append(impr)
    


# In[155]:


np.mean(overlap_rep_lin)


# In[46]:


segr_lin15 = cvResult(overlap_rate_mean_lin, coefdict_lin, 15, 0.7)


# In[47]:


(pd.DataFrame.from_dict(data=segr_lin15, orient='index')   .to_csv('segr_lin15_cmax.csv', header=False))


# In[233]:


from sklearn.linear_model import Ridge
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
    segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.7 ))
    Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
    intersect = len(Intersection(segr_ridge.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t, intersect)
    ii +=1


# In[ ]:





# In[244]:


segr_ridge15 = cvResult(overlap_rate_mean_ridge, coefdict_ridge, 15, 0.9)
Test_seg = BottomList(endal_seg, 'base_seg_id', 0.05)
len(Intersection(segr_ridge15.keys(), Test_seg))/len(Test_seg)


# In[240]:


len(Test_seg)


# In[243]:


len(segr_ridge15.keys())


# In[213]:


segkeys = [*segr_ridge12]
len(segkeys)


# In[214]:


len(Intersection(segr_ridge12.keys(), segr_ridge5.keys()))


# In[215]:


sedf= pd.DataFrame(list(segr_ridge12.items()),
                      columns=['Seg_id','Coef_value'])


# In[216]:


sedf.shape


# In[199]:


names_id = list(map(lambda x: re.findall('\d+', x), df_summary.base_seg_id))
flattened = [val for sublist in names_id for val in sublist]
df_summary['names_id'] = flattened


# In[200]:


cmax_seg_df = pd.merge(sedf, 
         df_summary, 
         how='left', left_on='Seg_id', right_on = 'names_id')


# In[201]:


cmax_seg_df.impressions


# In[206]:


cmax_seg_df_f = cmax_seg_df[(cmax_seg_df['marg_imp']>0) & (cmax_seg_df['cr'] <= 0.0003)]


# In[207]:


cmax_seg_df_f.shape


# In[208]:


rmseg = cmax_seg_df_f.base_seg_id


# In[209]:


improve_f(endal_1, rmseg)


# In[114]:


cmax_seg_df_f.to_csv('results/candidatecmaxall.csv', index = False)


# In[228]:


seg_suggest =  [*segr_ridge5][0:81] + nonsegid 
len(seg_suggest)


# In[229]:


test_lsn = len(df_camp)
overlap_rep_ridge = []
se = []
imp = []
est_seg_ridge = seg_suggest
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


# In[231]:


imp


# In[35]:


seg_bot= BottomList(endal_1,'base_seg_id', 0.1 )
len(seg_bot)


# In[224]:


test_lsn = len(camp_test_list)
overlap_rep_ridge = []
se = []
imp = []
est_seg_ridge = seg_suggest
for i in range (test_lsn):
    df_test = camp_test_list[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>=700:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
        ovlr = len(Intersection(est_seg_ridge, Test_seg_test))/len(Test_seg_test)
        se.append(segn)
        overlap_rep_ridge.append(ovlr)
        if df_test['cr'].mean()!=0:
            impr = improve_f(df_test,est_seg_ridge[0:int(segn*0.09)+1])
            #impr = improve_f(df_test,Test_seg_test)
            imp.append(impr)
    


# In[225]:


np.mean(overlap_rep_ridge)


# In[133]:


overlap_rep_ridge


# In[144]:


np.mean(imp)


# In[113]:


len(imp)


# In[40]:


len(Intersection(est_seg_ridge, est_seg_lin))


# In[41]:


(pd.DataFrame.from_dict(data=segr_ridge5, orient='index')   .to_csv('segr_ridge5_cmax.csv', header=False))


# In[ ]:


sedf= pd.DataFrame(list(segr_ridge5.items()),
                      columns=['Seg_id','Coef_value'])


# In[19]:


for train_indice, test_indice in ss.split(Train):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(Train.iloc[train_indice]['base_seg_id'].unique())


# In[65]:


pd.DataFrame(train_indice).to_csv("cmax_trainindice.csv")


# In[66]:


pd.DataFrame(test_indice).to_csv("cmax_testindice.csv")


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


# In[48]:


# Lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

lasso = Lasso(alpha=0.00000001)
lasso_y_pred = lasso.fit(X_train, y_train).predict(X_test) 
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_rmse
lasso_coef = lasso.coef_
lasso_coef
a_lasso = lasso_coef.argsort()
seg_lasso = CoefRank(X_train, a_lasso[0:bo10_segn+1])


# In[25]:


print(lasso_rmse, len(Intersection(seg_lasso, Test_seg_test))/len(Test_seg_test))


# In[26]:


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


# In[27]:


print(enet_rmse, len(Intersection(seg_enet, Test_seg_test))/len(Test_seg_test))


# In[29]:


# Benchmark: select the bottom 10% of the train data
Train_seg = BottomList(Train, 'base_seg_id', 0.1)
len(Intersection(Train_seg, Test_seg_test))/len(Test_seg_test)


# In[32]:


sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[31]:


import time
from datetime import timedelta


# In[37]:


import random
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


# In[38]:


index_max_bench = np.argmax(overlap_rate_mean)
seg_bench = seg_iteration[index_max_bench]
len(Intersection(Test_seg_test,seg_bench))/len(Test_seg_test)


# In[ ]:


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


# In[56]:


segr_lin15 = cvResult(overlap_rate_mean_lin, coefdict_lin, 10, 0.7 )
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_lin15.keys(), Test_seg_test))/len(Test_seg_test)


# In[53]:


len(Intersection(segr_lin.keys(), Test_seg_test))/len(Test_seg_test)


# In[54]:


len(Intersection(segr_lin10.keys(), Test_seg_test))/len(Test_seg_test)


# In[57]:


len(Intersection(segr_lin15.keys(), Test_seg_test))/len(Test_seg_test)


# In[46]:


(pd.DataFrame.from_dict(data=segr_lin10, orient='index')
   .to_csv('segr_lin10_cmax.csv', header=False))


# In[44]:


# segr_lin_20 = cvResult(overlap_rate_mean_lin, coefdict_lin, 20, 0.7 )
# Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
# len(Intersection(segr_lin_20.keys(), Test_seg_test))/len(Test_seg_test)


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


# In[ ]:





# In[61]:


segr_lasso = cvResult(overlap_rate_mean_lasso, coefdict_lasso, 6, 0.7 )
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_lasso.keys(), Test_seg_test))/len(Test_seg_test)


# In[ ]:


from sklearn.linear_model import Ridge
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


(pd.DataFrame.from_dict(data=segr_lasso, orient='index')
   .to_csv('segr_lasso6_cmax.csv', header=False))


# In[ ]:



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
    for train_indice, test_indice in sss.split(Train):
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


# In[46]:


segr_enet = cvResult(overlap_rate_mean_enet, coefdict_enet, 5, 0.7 )
Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_enet.keys(), Test_seg_test))/len(Test_seg_test)


# In[47]:


(pd.DataFrame.from_dict(data=segr_enet, orient='index')
   .to_csv('segr_enet6_cmax.csv', header=False))


# In[105]:


# test_lsn = len(camp_test_list)
# overlap_rep = []
# est_seg = segr.keys()
# for i in range (test_lsn):
#     df_test = camp_test_list[i]
#     Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
#     ovlr = len(Intersection(est_seg, Test_seg_test))/len(Test_seg_test)
#     overlap_rep.append(ovlr)


# In[106]:


#overlap_rep


# In[107]:


print(np.mean(overlap_rep), np.std(overlap_rep))


# In[94]:


[nonsegid, est_seg]


# In[37]:


R="\n".join(nonsegid)
f = open('Cinemax_noconversion.csv','w')
f.write(R)
f.close()


# In[ ]:




