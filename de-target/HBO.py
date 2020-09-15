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


# In[2]:



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
AND marketplace_id = 1
AND start_dt_utc > to_date('20190701','YYYYMMDD')
AND end_dt_utc < to_date('20190930','YYYYMMDD') 
ORDER BY dim_date ASC
LIMIT 1000000;

''', engine1)


# In[3]:


endal.apply(pd.Series.nunique)


# In[ ]:





# In[14]:


max(endal_1['end_dt_utc'] - endal_1['start_dt_utc'])


# In[4]:


endal_1 = endal.copy()
endal_1['conversions'].fillna(0, inplace = True)
endal_1['cr'] = endal_1['conversions']/endal_1['impressions']

print(endal_1['cr'].mean(), endal_1['cr'].std())


# In[5]:


print(endal_1['impressions'].mean(), endal_1['impressions'].std())


# In[13]:


endal_1['impressions'].quantile(0.5) 


# In[19]:


500000/7/1200


# In[6]:


print(endal_1['conversions'].mean(), endal_1['conversions'].std())


# In[5]:


df_summary = endal_1.groupby('base_seg_id').aggregate({'impressions':np.mean, 'cr': np.mean}).reset_index()


# In[6]:


df_summary.head(2)


# In[7]:


df_summary.to_csv('hbo_seg_info.csv', index= False)


# In[8]:


## segments that don't conversions across all records. So that they can be removed
segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.sum()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))
# Remove these segments from the data
seg_mask = endal_1['base_seg_id'].isin (nonsegid)
endal_seg = endal_1[~seg_mask]


# In[9]:


# split data by campaign id
gb = endal_seg.groupby('campaign_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*0)
camplist_test = shuffle[:camp_test_size].tolist()
camplist_train = shuffle[camp_test_size:].tolist()


# In[10]:


camp_train_list = [df_camp[i] for i in camplist_train]
camp_test_list = [df_camp[i] for i in camplist_test]
endal_train = pd.concat(camp_train_list)
#endal_test = pd.concat(camp_test_list)


# In[11]:


print('train.shape:', endal_train.shape)


# In[12]:


len(camplist_train)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


Train, Test = train_test_split (endal_train, test_size= 0.2, random_state = 42)


# In[15]:


from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1, test_size= 0.2, random_state = 6)


# In[16]:


endal_comp = endal_train[['base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape


# In[17]:


# onehot encoded seg_id
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix='seg')


# In[18]:


endal_encoded.head(2)


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

def improve_f(df, remove_id):
    test_data_remove = df[~df['base_seg_id'].isin(remove_id)]
    improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return improve * 100


# In[21]:


for train_indice, test_indice in ss.split(Train):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(Train.iloc[train_indice]['base_seg_id'].unique())


# In[22]:


train_indice


# In[64]:


#pd.DataFrame(train_indice).to_csv("hbo_trainindice.csv")


# In[65]:


#pd.DataFrame(test_indice).to_csv("hbo_testindice.csv")


# In[23]:


Test_seg = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
len(Test_seg)


# In[80]:


Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Test_seg_test)


# In[26]:


# Benchmark: select the bottom 10% of the train data
Train_seg = BottomList(Train, 'base_seg_id', 0.1)
len(Intersection(Train_seg, Test_seg_test))/len(Test_seg_test)


# In[27]:


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


# In[28]:


print(lin_rmse, len(Intersection(seg_lin, Test_seg_test))/len(Test_seg_test))


# In[30]:


# # Import necessary modules
# from sklearn.linear_model import RidgeCV
# from sklearn.model_selection import cross_val_score


# alpha_space = np.logspace(-8, 0, 10)
# reg_CV = RidgeCV(alphas=alpha_space, cv=5)
# reg_CV.fit(X_train, y_train)


# In[ ]:


# reg_CV.alpha_
# reg_CV.coef_


# In[31]:



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


# In[32]:


print(ridge_rmse, len(Intersection(seg_ridge, Test_seg_test))/len(Test_seg_test))


# In[ ]:


# from sklearn.linear_model import LassoCV
# from sklearn.model_selection import cross_val_score

# alpha_space = np.logspace(-8, 0, 5)
# lasso_CV = LassoCV(alphas=alpha_space, cv=5)
# lasso_CV.fit(X_train, y_train)


# In[ ]:


# lasso_CV.alpha_


# In[33]:


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


# In[34]:


print(lasso_rmse, len(Intersection(seg_lasso, Test_seg_test))/len(Test_seg_test))


# In[36]:


# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import GridSearchCV

# l1_space = np.linspace(0.1,1,10)
# alpha_space = np.logspace(-8, 0, 5)

# param_grid = {'l1_ratio': l1_space,
#              'alpha':alpha_space}

# elastic_net = ElasticNet()

# # Setup the GridSearchCV object: gm_cv
# gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# # Fit it to the training data
# gm_cv.fit(X_train, y_train)


# In[37]:


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


# In[38]:


print(enet_rmse, len(Intersection(seg_enet, Test_seg_test))/len(Test_seg_test))


# In[39]:


# Benchmark: select the bottom 10% of the train data
Train_seg = BottomList(Train, 'base_seg_id', 0.1)
len(Intersection(Train_seg, Test_seg_test))/len(Test_seg_test)


# In[40]:


import time
from datetime import timedelta


# In[41]:


sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[71]:



# Benchmark: select the bottom 10% of the train data as the prediction + Cross-validation
start_time = time.monotonic()
n = 15
seg_iteration = []
overlap_rate_mean = []
imp_mean = []
ii = 1
random.seed(30)
sss = ShuffleSplit(n_splits=5, test_size= 0.2)
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
    k = {k:v for k, v in c.items() if v >= int(0.7* min(i, 9))}
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


# In[ ]:





# In[ ]:





# In[65]:





# In[ ]:





# In[72]:


index_max_bench = np.argmax(overlap_rate_mean)
seg_bench = seg_iteration[index_max_bench]
len(Intersection(Test_seg_test,seg_bench))/len(Test_seg_test)


# In[ ]:


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
        train_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        test_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
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
    Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
    intersect = len(Intersection(segr_lin.keys(), Test_seg_test))/len(Test_seg_test)
    print (ii, delta_t, intersect)
    ii +=1


# In[78]:


segr_lin10 = cvResult(overlap_rate_mean_lin, coefdict_lin, 5, 0.7)
len(segr_lin10)


# In[81]:



#Test_seg_test = BottomList(Test, 'base_seg_id', 0.1)
len(Intersection(segr_lin10.keys(), Test_seg_test))/len(Test_seg_test)


# In[ ]:





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
    for train_indice, test_indice in sss.split(Train):
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
        train_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        test_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
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


# In[68]:



# Cross validation for Ridge regression
start_time = time.monotonic()
n = 5

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
        train_boid = BottomList(Train.iloc[train_indice], 'base_seg_id', 0.1)
        test_boid = BottomList(Train.iloc[test_indice], 'base_seg_id', 0.1)
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

