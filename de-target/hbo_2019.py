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


# In[2]:


segment_id = pd.read_sql_query(''' 

SELECT distinct base_seg_id
FROM VAP_AGG.SEG_PVC_PERFORMANCE pf
WHERE pf.advertiser_name = 'HBO - Television - US'

''', engine1)


# In[5]:


segment_id.shape


# In[ ]:


df1.groupby()


# In[103]:


272/6744


# In[ ]:





# In[4]:


dat = pd.read_sql_query(''' 

SELECT *
FROM VAP_AGG.SEG_PVC_PERFORMANCE st
WHERE st.advertiser_name = 'HBO - Television - US'

''', engine1)


# In[65]:


dat.shape


# In[66]:


dat.columns


# In[67]:


dat['conversions'].sum()


# In[68]:


dat.ad_cfid.nunique()


# In[69]:


#2692+773


# In[70]:


dat.campaign_cfid.nunique()


# In[71]:


endal_1 = dat.copy()
endal_1['conversions'].fillna(0, inplace = True)
endal_1['cr'] = endal_1['conversions']/endal_1['impressions']

print(endal_1['cr'].mean(), endal_1['cr'].std())


# In[72]:


gb = endal_1.groupby(['campaign_cfid','ad_cfid'])['cr'].agg(['mean','std']).reset_index()
nonad = gb[gb['mean'] == 0].ad_cfid


# In[73]:


len(nonad)


# In[74]:


600/2692


# In[75]:


## segments that don't conversions across all records. So that they can be removed
segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.sum()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))
# Remove these segments from the data
#seg_mask = endal_1['base_seg_id'].isin (nonsegid)
#endal_seg = endal_1[~seg_mask]


# In[76]:


from functools import reduce
from collections import Counter

def BottomList (dataframe, colname, pct):
    groupby_df = dataframe.groupby(colname, as_index=False)['cr'].agg(['mean', 'count']).sort_values('mean',ignore_index=False).reset_index()
    smalln = math.ceil(len(groupby_df)*pct)
    seg_removed_bottom = groupby_df.nsmallest(smalln,'mean')[colname].to_list()
    bottom_segid = [i for i in seg_removed_bottom]
    return bottom_segid

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  


def CoefRank (df, ranka):
    names = list(df.columns[ranka].values)
    result = list(map(lambda x: remove_prefix(x,'seg_'), names))
    return(result)

def CoefNeg (df, coef):
    index = [i for i, e in enumerate(coef) if e < 0]
    names = list(df.columns[index].values)
    #result = list(map(lambda x: re.findall('\d+', x), names))
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


def ResultComple (dic_result, DF_summary):
    sedf= pd.DataFrame(list(dic_result.items()),
                      columns=['Seg_id','Coef_value'])
    
    merged_seg_df = pd.merge(sedf, 
         DF_summary, 
         how='left', left_on='Seg_id', right_on = 'base_seg_id').sort_values (by = 'Coef_value')
    return merged_seg_df


# In[77]:


# def improve_f(df, remove_id):
#     test_data_remove = df[~df['base_seg_id'].isin(remove_id)]
#     avg_new = np.sum(test_data_remove['conversions'])/np.sum(test_data_remove['impressions'])
#     avg_orig = np.sum(df['conversions'])/np.sum(df['impressions'])
#     improve = (avg_new - avg_orig)/avg_orig
#     #improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
#     return improve * 100

def improve_cps (df, remove_id):
    tt = df[(df.cr !=0)]
    cps = tt.impressions/tt.cr
    test_data_remove = df[(~df['base_seg_id'].isin(remove_id)) & (df.cr !=0)]
    new_cps = np.mean(test_data_remove.impressions/test_data_remove.cr)
    imp = (new_cps - cps)/cps

def impression_de(df, remove_id):
    test_data_remove = df[~df['base_seg_id'].isin(remove_id)]
    sum_new = np.sum(test_data_remove['impressions'])
    sum_orig = np.sum(df['impressions'])
    decrease = (sum_new - sum_orig)/sum_orig
    #improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return decrease * 100

def marg_improve (df, remove_id):
    is_removed = df['base_seg_id']== remove_id
    df_new = df[~is_removed]
    avg_new = np.sum(df_new['conversions'])/np.sum(df_new['impressions'])
    avg_orig = np.sum(df['conversions'])/np.sum(df['impressions'])
    imp = (avg_new - avg_orig)/avg_orig
    return imp


# In[78]:


df_summary = endal_1.groupby('base_seg_id').aggregate({'impressions':np.mean, 
                                                       'cr': np.mean,
                                                      'conversions': np.mean}).reset_index()

marg_imp = [marg_improve (df_summary, i) for i in df_summary.base_seg_id]
df_summary['marg_imp'] = marg_imp
df_summary.shape


# In[79]:


df_summary.marg_imp.mean()


# In[80]:


np.mean(df_summary.cr)


# In[81]:


df_impressionmax = endal_1.groupby('base_seg_id').aggregate({'impressions':np.max, 
                                                       'cr': np.mean
                                                      }).reset_index()


# In[ ]:





# In[104]:


test_data_remove = df_summary[(~df_summary['base_seg_id'].isin(k1)) & (df_summary.cr !=0)]


# In[122]:


np.mean(test_data_remove.impressions/test_data_remove.conversions)


# In[106]:


tt1 = df_summary[(df_summary.cr !=0)]


# In[123]:


np.mean(tt1.impressions/tt1.conversions)


# In[124]:


(np.mean(test_data_remove.impressions) - np.mean(tt1.impressions))/np.mean(test_data_remove.impressions)


# In[115]:


(np.mean(test_data_remove.conversions) - np.mean(tt1.conversions))/np.mean(test_data_remove.conversions)


# In[116]:


(np.mean(test_data_remove.cr) - np.mean(tt1.cr))/np.mean(test_data_remove.cr)


# In[118]:


1- 0.96/1.07


# In[111]:


(3144-1914)/3144


# In[121]:





# In[112]:


(4965038.279979611-4284934.447505558)/4965038.279979611


# In[82]:


df_s2 = endal_1['base_seg_id'].value_counts().reset_index()
df_s3 = pd.merge(df_summary, df_s2[['index', 'base_seg_id']], left_on = 'base_seg_id', right_on = 'index')
df_s3['count'] = df_s3['base_seg_id_y']
df_s3['base_seg_id'] = df_s3['base_seg_id_x']
df_s3 = df_s3.drop(['index','base_seg_id_y','base_seg_id_x'], axis = 1)
df_s3 = df_s3.sort_values(by = 'cr')


# In[83]:


df_s3_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0) ]
#df_s3_cand.to_csv('data_2019/cmax_cand.csv')
seg_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0)].base_seg_id.tolist()


# In[84]:


len(seg_cand)


# In[50]:


#df_s3_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0) & (df_s3['count'] > 1000) ]
#df_s3_cand ['ad_name'] = 'hbo'


# In[34]:


df_s3_cand.to_csv('data_2019/hbo_cand.csv')


# In[ ]:





# In[85]:


imp_mean = df_s3.marg_imp.mean()
imp_mean


# In[36]:


#len(k1)


# In[37]:


#270/6744


# In[328]:


#seg_cand


# In[86]:


gb = endal_1.groupby(['campaign_cfid','ad_cfid'])['cr'].agg(['mean','std']).reset_index()
nonad = gb[gb['mean'] == 0].ad_cfid

seg_mask = endal_1['ad_cfid'].isin (nonad)


# In[87]:


endal_mask1 = endal_1[~seg_mask]
endal_seg = endal_mask1[(endal_mask1['base_seg_id'].isin (seg_cand))]


# In[88]:


endal_seg.base_seg_id.nunique()


# In[89]:


endal_seg.shape


# In[90]:


df_selected = endal_seg


# In[290]:


# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size= 0.8, random_state = 42)
# for selected_indice, rest_indice in split.split(endal_seg, endal_seg['base_seg_id']):
#     df_selected = endal_seg.iloc[selected_indice]
#     df_rest = endal_seg.iloc[rest_indice]


# In[91]:


df_selected.shape


# In[26]:


#import xlsxwriter


# In[28]:


#endal_seg.to_excel('data_2019/hbo_df_filter.xlsx', engine='xlsxwriter')


# In[430]:


#endal_seg.ad_cfid.nunique()


# In[92]:


endal_comp = df_selected[[ 'base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape


# In[166]:





# In[59]:


df_selected.columns


# In[60]:


df_selected.cr.std()


# In[61]:


df_selected.cr.mean()


# In[62]:


1 - abs((0.0027931 - 0.0028055)/0.0027931)


# In[93]:


# onehot encoded seg_id
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix=['seg'])


# In[94]:


endal_encoded.shape


# In[48]:


len(endal_encoded.columns)


# In[95]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[50]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)
for train_indice, test_indice in sss.split(endal_comp, endal_comp['base_seg_id']):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(endal_comp.iloc[train_indice]['base_seg_id'].unique())


# In[51]:


# Test_seg_test = BottomList(df_rest, 'base_seg_id', 0.1)
# len(Test_seg_test)


# In[52]:


# Linear regression 
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_coef = lin_reg.coef_
y_pred = lin_reg.predict(X_test) 
train_predictions = lin_reg.predict(X_train)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
a_lin = lin_coef.argsort()
bo10_segn = int(seg_n*0.1)
seg_lin = CoefRank(X_train, a_lin[0:bo10_segn+1])
 


# In[53]:


print(lin_rmse)


# In[54]:


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
seg_ridge = CoefRank(X_train, a_ridge[0:20])


# In[55]:


print(ridge_rmse)


# In[56]:


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
print(lasso_rmse)


# In[57]:


from sklearn.linear_model import ElasticNet

alpha = 0.000000001
enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
enet_y_pred = enet.fit(X_train, y_train).predict(X_test)
enet_mse = mean_squared_error(y_test, enet_y_pred)
enet_rmse = np.sqrt(enet_mse)
enet_rmse
enet_coef  = enet.coef_
a_enet = enet_coef.argsort()
seg_enet = CoefRank(X_train, a_enet[0:bo10_segn+1])


# In[ ]:


print(enet_rmse)


# In[441]:


X_train.columns


# In[180]:


#a_ridge


# In[370]:


index = [i for i, x in enumerate(ridge_coef) if x < 0]


# In[371]:


len(index)


# In[ ]:





# In[390]:


ridge_coef[index]
names = X_train.columns[index].values
seg_id = [remove_prefix(i, 'seg_') for i in names]
len(seg_id)


# In[64]:


seg_id


# In[408]:


# seg_id_n = [x for x in seg_id if not x.startswith('ad_')]
# len(seg_id_n)


# In[96]:


# Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Cross validation for Ridge regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_ridge = []
imp_mean_ridge = []
rmse_mean_ridge = []
coefdict_ridge = []
ii = 1
alpha = 1
random.seed(30)
for i in range (n):
    
    overlap_rate = []
    imp_rate= []
    coef_dic = []
    rmse = []
    
    for train_indice, test_indice in sss.split(endal_comp):
        df_train = endal_encoded.iloc[train_indice]
        df_test  = endal_encoded.iloc[test_indice]
        X_train = df_train.drop('cr', axis=1)
        y_train = df_train.cr
        X_test = df_test.drop('cr', axis = 1)
        y_test = df_test.cr
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_y_pred  = ridge.fit(X_train, y_train).predict(X_test)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_y_pred))
        list_coef = ridge.coef_
        test_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
        #a = list_coef.argsort()
        index = [i for i, x in enumerate(list_coef) if x < 0]
        coefcollect = list_coef[index]
        ml_segid = CoefRank(X_train, index)
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        imp = improve_f(df_selected.iloc[test_indice], ml_segid)
        #test_com = endal_seg.iloc[test_indice]
        #imp = AdTest(test_com, ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
        rmse.append(ridge_rmse)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_ridge.append(dd)
    overlap_rate_mean_ridge.append(np.mean(overlap_rate))
    imp_mean_ridge.append(np.mean(imp_rate))
    rmse_mean_ridge.append (np.mean(rmse)) 
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    #segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.9 ))
    #Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
    #intersect = len(Intersection(segr_ridge.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t)
    ii +=1


# In[97]:


segr_ridge15 = cvResult(imp_mean_ridge, coefdict_ridge, 15,1)
len(segr_ridge15)


# In[ ]:





# In[98]:


segr_ridge15


# In[99]:


k1 = sorted(segr_ridge15, key=segr_ridge15.get)


# In[100]:


sed = k1


# In[166]:


df_summary[df_summary.base_seg_id.isin(k1)].cr.mean()


# In[167]:


df_summary[~df_summary.base_seg_id.isin(k1+ nonsegid) ].cr.mean()


# In[355]:


def ResultComple (dic_result, DF_summary):
    sedf= pd.DataFrame(list(dic_result.items()),
                      columns=['Seg_id','Coef_value'])
    
    merged_seg_df = pd.merge(sedf, 
         DF_summary, 
         how='left', left_on='Seg_id', right_on = 'base_seg_id').sort_values (by = 'Coef_value')
    return merged_seg_df


# In[101]:


segid_df = ResultComple(segr_ridge15, df_s3)


# In[95]:


segid_df[0:50].impressions.sum()


# In[97]:


70596/1347899


# In[372]:


segid_df['ad_name'] = 'hbo'


# In[102]:


segid_df.to_csv ('data_2019/hbo_detarget00.csv', index= False)


# In[524]:


seg_ridge


# In[526]:


from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.5)


# In[32]:


overlap_rate = []
imp_rate= []
coef_dic = []
rmse = []
alpha = 1
for train_indice, test_indice in ss.split(df_selected):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_y_pred))
    list_coef = ridge.coef_
    test_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.3)
    train_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.3)
    index = [i for i, x in enumerate(list_coef) if x < 0]
    coefcollect = list_coef[index]
    ml_segid = CoefRank(X_train, index)
    coefdd = dict(zip(ml_segid, coefcollect))
    intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
    imp = improve_f(endal_seg.iloc[test_indice], ml_segid)
    coef_dic.append(coefdd)
    overlap_rate.append(intersect)
    imp_rate.append(imp)
    rmse.append(ridge_rmse)


# In[ ]:





# In[198]:


remove_prefix('seg_x9424016011', 'seg_')


# In[199]:


from sklearn.ensemble import RandomForestRegressor 

# Instantiate rf
rf = RandomForestRegressor(n_estimators=4,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 


# In[200]:


rf_y_pred = rf.predict (X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_rmse


# In[203]:


importances_dr = pd.Series(rf.feature_importances_, index = X_train.columns)
sorted_importance_dr = importances_dr.sort_values(ascending = False).reset_index()
#sorted_importance_dr


# In[269]:


seg_id = [remove_prefix(i, 'seg_') for i in sorted_importance_dr['index']]


# In[270]:


seg_id


# In[73]:


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
VAP_AGG.ENDAL_PVC_SEGMENTS_TRAFFIC st
LEFT JOIN VAP_AGG.ENDAL_PVC_SEGMENTS_CONV sc
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
AND DATEDIFF(day, st.start_dt_utc, st.dim_date) > 14
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


# In[ ]:





# In[74]:


endal_10 = endal_test.copy()
endal_10['conversions'].fillna(0, inplace = True)
endal_10['cr'] = endal_10['conversions']/endal_10['impressions']

print(endal_10['cr'].mean(), endal_10['cr'].std())


# In[75]:


# split data by ad id
gb = endal_10.groupby('ad_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]


# In[399]:


suggestseg = seg_cand[0:335]


# In[342]:


len(suggestseg)


# In[77]:


suggestseg = sed
len(suggestseg)


# In[306]:


#suggestseg


# In[156]:


test_lsn = len(df_camp)
overlap_rep_ridge = []
se = []
itp = []
imp = []
cps = []
decrease_impression = []
est_seg_ridge = suggestseg
for i in range (test_lsn):
    df_test = df_camp[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>1000:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
        ovlr = len(Intersection(est_seg_ridge, Test_seg_test))
        se.append(segn)
        decr = impression_de(df_test,est_seg_ridge )
        overlap_rep_ridge.append(ovlr)
        if df_test['cr'].mean()!=0:
            impr = improve_f(df_test,est_seg_ridge)
            impr_it = improve_f(df_test,Test_seg_test)
            imp_cps = improve_cps (df_test, est_seg_ridge)
            imp.append(impr)
            itp.append(impr_it)
            cps.append(imp_cps)
            decrease_impression.append(decr)


# In[181]:


np.mean(imp)


# In[182]:


np.mean(itp)


# In[169]:


res


# In[178]:


np.mean(imp)


# In[129]:





# In[179]:


np.mean(itp)


# In[81]:


np.mean(decrease_impression)


# In[123]:


len(imp)


# In[ ]:




