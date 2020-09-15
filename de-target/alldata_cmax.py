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
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
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


# In[ ]:





# In[2]:


dat_cmax = pd.read_sql_query(''' 

SELECT *
FROM VAP_AGG.SEG_PVC_PERFORMANCE st
WHERE st.advertiser_name = 'Cinemax - US'

''', engine1)


# In[3]:


dat_cmax.base_seg_id.nunique()


# In[5]:


dat_cmax.campaign_cfid.nunique()


# In[4]:


#dat_cmax.start_dt_utc.unique()


# In[14]:


#dat_cmax.to_json('data_2019/cmax.json')


# In[6]:


endal_1 = dat_cmax.copy()
endal_1['conversions'].fillna(0, inplace = True)
endal_1['cr'] = endal_1['conversions']/endal_1['impressions']

print(endal_1['cr'].mean(), endal_1['cr'].std())


# In[7]:


endal_1['ad_cfid'].nunique()


# In[ ]:





# In[8]:


## segments that don't conversions across all records. So that they can be removed
segg = endal_1.groupby('base_seg_id')['cr']
sumg = segg.sum()
nonsegid = sumg[sumg == 0].reset_index()['base_seg_id'].tolist()
print('Number of none conversion segments:', len(nonsegid))
# Remove these segments from the data
#seg_mask = endal_1['base_seg_id'].isin (nonsegid)
#endal_seg = endal_1[~seg_mask]


# In[9]:


gb = endal_1.groupby(['campaign_cfid','ad_cfid'])['cr'].agg(['mean','std']).reset_index()
nonad = gb[gb['mean'] == 0].ad_cfid
len(nonad)


# In[10]:


seg_mask = endal_1['ad_cfid'].isin (nonad)
endal_mask1 = endal_1[~seg_mask]


# In[11]:


endal_mask1.shape


# In[ ]:





# In[ ]:





# In[12]:


#ad_segn = endal_1.groupby(['ad_cfid'])['base_seg_id'].nunique()


# In[13]:


from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[14]:


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
    avg_new = np.sum(test_data_remove['conversions'])/np.sum(test_data_remove['impressions'])
    avg_orig = np.sum(df['conversions'])/np.sum(df['impressions'])
    improve = (avg_new - avg_orig)/avg_orig
    #improve = (test_data_remove['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return improve * 100

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

def marg_improve2 (df, remove_id):
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

# def ResultComple (dic_result, DF_summary):
#     sedf= pd.DataFrame(list(dic_result.items()),
#                       columns=['Seg_id','Coef_value'])
#     names_id = list(map(lambda x: re.findall('\d+', x), DF_summary.base_seg_id))
#     flattened = [val for sublist in names_id for val in sublist]
#     DF_summary['names_id'] = flattened
#     marg_imp = [marg_improve (DF_summary, i) for i in DF_summary.base_seg_id]
#     DF_summary['marg_imp'] = marg_imp
#     merged_seg_df = pd.merge(sedf, 
#          DF_summary, 
#          how='left', left_on='Seg_id', right_on = 'names_id')
#     merged_seg_df_f = merged_seg_df[(merged_seg_df['marg_imp']>0)].sort_values(by='Coef_value', ascending=True)
#     nonseg_df = DF_summary[(DF_summary.cr == 0)].sort_values(by='impressions', ascending=True)
#     return merged_seg_df_f, nonseg_df


# In[15]:


def adTest (df, seg):
    gb = df.groupby('ad_cfid')    
    df_camp = [gb.get_group(x) for x in gb.groups]
    test_lsn = len(df_camp)
    overlap = []
    se = []
    itp = []
    imp = []
    est_seg_ridge = seg
    for i in range (test_lsn):
        df_test = df_camp[i]
        segn = len(df_test['base_seg_id'].unique())
        if segn>1000:
            Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
            ovlr = len(Intersection(est_seg_ridge, Test_seg_test))/len(Test_seg_test)
            se.append(segn)
            decr = impression_de(df_test,est_seg_ridge )
            overlap.append(ovlr)
            if df_test['cr'].mean()!=0:
                impr = improve_f(df_test,est_seg_ridge)
                impr_it = improve_f(df_test,Test_seg_test)
                imp.append(impr)
                itp.append(impr_it)
                decrease_impression.append(decr)
    return np.mean(imp), np.mean(overlap)
        
    
    
    


# In[16]:


df_summary = endal_mask1.groupby('base_seg_id').aggregate({'impressions':np.mean, 
                                                       'cr': np.mean,
                                                      'conversions': np.mean}).reset_index()
df_summary.head(2)


# In[ ]:


# def marg_improve (df, remove_id):
#     is_removed = df['base_seg_id']== remove_id
#     df_new = df[~is_removed]
#     avg_new = np.sum(df_new['conversions'])/np.sum(df_new['impressions'])
#     avg_orig = np.sum(df['conversions'])/np.sum(df['impressions'])
#     imp = (avg_new - avg_orig)/avg_orig
#     return imp


# In[17]:


marg_imp = [marg_improve2 (df_summary, i) for i in df_summary.base_seg_id]


# In[ ]:





# In[18]:


df_summary['marg_imp'] = marg_imp
df_summary.shape


# In[19]:


df_s2 = endal_1['base_seg_id'].value_counts().reset_index()
df_s3 = pd.merge(df_summary, df_s2[['index', 'base_seg_id']], left_on = 'base_seg_id', right_on = 'index')
df_s3['count'] = df_s3['base_seg_id_y']
df_s3['base_seg_id'] = df_s3['base_seg_id_x']
df_s3 = df_s3.drop(['index','base_seg_id_y','base_seg_id_x'], axis = 1)
df_s3 = df_s3.sort_values(by = 'cr')


# In[840]:


#df_s3 = pd.merge(df_summary, df_s2[['index', 'base_seg_id']], left_on = 'base_seg_id', right_on = 'index')


# In[841]:


#df_s3['base_seg_id'] = df_s3['base_seg_id_x']
#df_s3 = df_s3.sort_values(by = 'cr')


# In[843]:


#len(df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0) & (df_s3['count'] > 1000) ].base_seg_id.tolist() )      


# In[20]:


df_s3_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0)]
#df_s3_cand.to_csv('data_2019/cmax_cand.csv')


# In[21]:


seg_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0) ].base_seg_id.tolist()


# In[943]:


#df_s3['count'] = df_s3['base_seg_id_y']
#df_s3['base_seg_id'] = df_s3['base_seg_id_x']
#df_s3 = df_s3.drop(['index','base_seg_id_y'], axis = 1)


# In[22]:


len(seg_cand)


# In[945]:


#df_s3.sort_values(by = 'cr').to_csv('data_2019/cmax_seg_info.csv', index= False)


# In[23]:


imp_mean = df_s3.marg_imp.mean()
imp_mean


# In[ ]:





# In[24]:


len(seg_cand)


# In[25]:


df_s3_cand = df_s3[(df_s3.cr !=0) & (df_s3.marg_imp > 0)]
#df_s3_cand['ad_name'] = 'cinemax'


# In[26]:


df_s3_cand.to_csv('data_2019/cinemax_cand.csv')


# In[27]:


gb = endal_1.groupby(['campaign_cfid','ad_cfid'])['cr'].agg(['mean','std']).reset_index()
nonad = gb[gb['mean'] == 0].ad_cfid

seg_mask = endal_1['ad_cfid'].isin (nonad)
endal_mask1 = endal_1[~seg_mask]


# In[266]:


# split data by campaign id
gb_ad = endal_mask1.groupby('ad_cfid')    
df_camp_ad = [gb_ad.get_group(x) for x in gb_ad.groups]


# In[ ]:





# In[28]:


endal_seg = endal_mask1[(endal_mask1['base_seg_id'].isin (seg_cand))]


# In[29]:


#endal_seg.base_seg_id.nunique()


# In[30]:


endal_seg.shape


# In[31]:


endal_seg.base_seg_id.nunique()


# In[32]:


df_selected = endal_seg 


# In[33]:


# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size= 0.8, random_state = 42)
# for selected_indice, rest_indice in split.split(endal_seg, endal_seg['base_seg_id']):
#     df_selected = endal_seg.iloc[selected_indice]
#     df_rest = endal_seg.iloc[rest_indice]


# In[34]:


#import xlsxwriter


# In[28]:


#endal_seg.to_excel('data_2019/cmax_df_filter.xlsx', engine='xlsxwriter')


# In[35]:


# split data by campaign id
gb = endal_seg.groupby('ad_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]
shuffle = np.random.permutation (len(df_camp))
camp_test_size = int(len(df_camp)*0)
camplist_test = shuffle[:camp_test_size].tolist()
camplist_train = shuffle[camp_test_size:].tolist()


# In[36]:


camp_train_list = [df_camp[i] for i in camplist_train]
camp_test_list = [df_camp[i] for i in camplist_test]


# In[37]:


endal_comp = df_selected[[ 'base_seg_id', 'cr']]
endal_comp['const'] = 1
endal_comp.shape


# In[40]:


#11551350*0.2


# In[41]:


#endal_comp.campaign_cfid.nunique()


# In[ ]:





# In[101]:


df_selected['base_seg_id'].nunique()


# In[43]:


# onehot encoded seg_id
endal_encoded = pd.get_dummies(endal_comp, columns=['base_seg_id'], prefix=['seg'])


# In[44]:


endal_encoded.shape


# In[84]:


from sklearn.model_selection import StratifiedShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size= 0.2, random_state = 4)


# In[73]:


for train_indice, test_indice in ss.split(endal_comp, endal_comp['base_seg_id']):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(endal_comp.iloc[train_indice]['base_seg_id'].unique())


# In[74]:


#Test_seg_test = BottomList(df_rest, 'base_seg_id', 0.1)
#len(Test_seg_test)


# In[75]:


df_selected.shape


# In[92]:


df_selected.columns


# In[93]:


df_selected.cr.std()


# In[98]:


1- abs((0.0016827 - 0.0016859)/0.0016827)


# In[94]:


df_selected.cr.mean()


# In[76]:


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
print(lin_rmse, len(Intersection(seg_lin, Test_seg_test))/len(Test_seg_test))


# In[821]:


# Import necessary modules
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


alpha_space = np.logspace(-8, 0, 10)
reg_CV = RidgeCV(alphas=alpha_space, cv=5)
reg_CV.fit(X_train, y_train)


# In[822]:


reg_CV.alpha_


# In[77]:


# Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ridge = Ridge(alpha= 1)
ridge_y_pred = ridge.fit(X_train, y_train).predict(X_test) 
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_rmse
ridge_coef = ridge.coef_
a_ridge = ridge_coef.argsort()
seg_ridge = CoefRank(X_train, a_ridge)


# In[78]:


print(ridge_rmse)


# In[79]:


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


# In[81]:


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


# In[83]:


print(enet_rmse)


# In[538]:


index = [i for i, x in enumerate(ridge_coef) if x < 0]


# In[539]:


ridge_coef[index]
names = X_train.columns[index].values
seg_id = [remove_prefix(i, 'seg_') for i in names]


# In[540]:


seg_id_n = [x for x in seg_id if not x.startswith('ad_')]
len(seg_id_n)


# In[268]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)
for train_indice, test_indice in sss.split(endal_comp, endal_comp['ad_cfid']):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(endal_comp.iloc[train_indice]['base_seg_id'].unique())


# In[487]:


test = endal_seg.iloc[test_indice]


# In[488]:


gbtest = test.groupby('ad_cfid')    
df_camptest = [gbtest.get_group(x) for x in gbtest.groups]


# In[498]:


df_camptest[9].shape


# In[723]:


endal_comp.cr.std()


# In[960]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size= 0.2)


# In[45]:



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
        #imp = adTest (endal_10,ml_segid)[0]
        #intersect = adTest (endal_10,ml_segid)[1]
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
    print (ii, delta_t, np.mean(imp_rate),np.mean(rmse),np.mean(overlap_rate) )
    ii +=1


# In[105]:


segr_ridge15 = cvResult(imp_mean_ridge, coefdict_ridge, 15, 1)
len(segr_ridge15)


# In[47]:


imp_mean_ridge


# In[48]:


rmse_mean_ridge


# In[86]:


df_per = pd.DataFrame(list(zip(imp_mean_ridge, rmse_mean_ridge, overlap_rate_mean_ridge)), 
               columns =['imp', 'rmse', 'overlap'])


# In[87]:


#df_per


# In[88]:


segr_ridge15 = cvResult(imp_mean_ridge, coefdict_ridge, 15, 1)
len(segr_ridge15)


# In[90]:


#segr_ridge15


# In[89]:


k1 = sorted(segr_ridge15, key=segr_ridge15.get)


# In[106]:


k1


# In[51]:


# from collections import OrderedDict
# od1 = OrderedDict(sorted(segr_ridge15.items(), key=lambda t: t[1]))
# od1


# In[52]:


seid = k1


# In[53]:


df_summary[df_summary.base_seg_id.isin(seid)].cr.mean()


# In[54]:


df_summary[~df_summary.base_seg_id.isin(seid+nonsegid)].cr.mean()


# In[107]:


segid_df = ResultComple(segr_ridge15, df_s3)


# In[ ]:


def marg_improve2 (df, remove_id):
    is_removed = df['base_seg_id']== remove_id
    df_new = df[~is_removed]
    imp = (df_new['cr'].mean()-df['cr'].mean())/df['cr'].mean()
    return imp


# In[765]:


#segid_df['ad_name'] = 'cinemax'


# In[109]:


segid_df.to_csv ('data_2019/cinemax_detarget00.csv', index= False)


# In[485]:


#segr_ridge15


# In[921]:


seg_id = [*segr_ridge15]


# In[531]:


endal_encoded.shape


# In[ ]:


endal_comp


# In[576]:


overlap_rate = []
imp_rate= []
coef_dic = []
rmse = []
for train_indice, test_indice in ss.split(endal_comp):
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
    test_boid = BottomList(endal_comp.iloc[train_indice], 'base_seg_id', 0.3)
    train_boid = BottomList(endal_comp.iloc[test_indice], 'base_seg_id', 0.3)
    index = [i for i, x in enumerate(list_coef) if x < 0]
    coefcollect = list_coef[index]
    ml_segid = CoefRank(X_train, index)
    #coefcollect = list_coef[a[0:len(train_boid)]]
    #ml_segid = CoefRank(X_train, a[0:len(train_boid)])
    coefdd = dict(zip(ml_segid, coefcollect))
    intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
    imp = improve_f(df_selected.iloc[test_indice], ml_segid)
    #imp = adTest (df_selected.iloc[test_indice], ml_segid)
    coef_dic.append(coefdd)
    overlap_rate.append(intersect)
    imp_rate.append(imp)
    rmse.append(ridge_rmse)


# In[577]:


imp_rate


# In[295]:


dd = defaultdict(list)
for d in coef_dic:
    for key, value in d.items():
        dd[key].append(value)
            
dd = dict(dd)


# In[ ]:





# In[99]:



avgDict = {}
for k,v in dd.items():
    avgDict[k] = sum(v)/ float(len(v))

len(avgDict)


# In[310]:


seg_id_n = [*avgDict]


# In[ ]:


for train_indice, test_indice in ss.split(endal_comp, endal_comp['base_seg_id']):
    df_train = endal_encoded.iloc[train_indice]
    df_test  = endal_encoded.iloc[test_indice]
    X_train = df_train.drop('cr', axis=1)
    y_train = df_train.cr
    X_test = df_test.drop('cr', axis = 1)
    y_test = df_test.cr
    seg_n = len(endal_comp.iloc[train_indice]['base_seg_id'].unique())


# In[816]:


from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

alpha_space = np.logspace(-8, 0, 5)
lasso_CV = LassoCV(alphas=alpha_space, cv=5)
lasso_CV.fit(X_train, y_train)


# In[817]:


lasso_CV.alpha_


# In[820]:


len(lasso_CV.coef_[lasso_CV.coef_<0])


# In[ ]:



# Cross validation for Lasso regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_lasso = []
imp_mean_lasso = []
rmse_mean_lasso = []
coefdict_lasso = []
ii = 1
alpha = 0.000000001
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
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        lasso_y_pred  = lasso.fit(X_train, y_train).predict(X_test)
        lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_y_pred))
        list_coef = lasso.coef_
        test_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
        #a = list_coef.argsort()
        index = [i for i, x in enumerate(list_coef) if x < 0]
        coefcollect = list_coef[index]
        ml_segid = CoefRank(X_train, index)
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        #imp = improve_f(df_selected.iloc[test_indice], ml_segid)
        #test_com = endal_seg.iloc[test_indice]
        imp = adTest(endal_10, ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
        rmse.append(lasso_rmse)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_lasso.append(dd)
    overlap_rate_mean_lasso.append(np.mean(overlap_rate))
    imp_mean_lasso.append(np.mean(imp_rate))
    rmse_mean_lasso.append (np.mean(rmse)) 
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    #segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.9 ))
    #Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
    #intersect = len(Intersection(segr_ridge.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t, np.mean(imp_rate),np.mean(rmse),np.mean(overlap_rate) )
    ii +=1


# In[ ]:





# In[ ]:



# Cross validation for Elastic regression
start_time = time.monotonic()
n = 15

overlap_rate_mean_enet = []
imp_mean_enet = []
rmse_mean_enet = []
coefdict_enet = []
ii = 1
alpha = 0.000000001
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
        enet = ElasticNet(alpha=alpha, l1_ratio=0.9)
        enet.fit(X_train, y_train)
        enet_y_pred  = enet.fit(X_train, y_train).predict(X_test)
        enet_rmse = np.sqrt(mean_squared_error(y_test, enet_y_pred))
        list_coef = enet.coef_
        test_boid = BottomList(endal_seg.iloc[train_indice], 'base_seg_id', 0.1)
        train_boid = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.1)
        #a = list_coef.argsort()
        index = [i for i, x in enumerate(list_coef) if x < 0]
        coefcollect = list_coef[index]
        ml_segid = CoefRank(X_train, index)
        coefdd = dict(zip(ml_segid, coefcollect))
        intersect = len(Intersection(test_boid, ml_segid))/len(test_boid)
        #imp = improve_f(df_selected.iloc[test_indice], ml_segid)
        #test_com = endal_seg.iloc[test_indice]
        imp = adTest(endal_10, ml_segid)
        coef_dic.append(coefdd)
        overlap_rate.append(intersect)
        imp_rate.append(imp)
        rmse.append(enet_rmse)
    
    dd = defaultdict(list)
    for d in coef_dic:
        for key, value in d.items():
            dd[key].append(value)
            
    dd = dict(dd)
    coefdict_enet.append(dd)
    overlap_rate_mean_enet.append(np.mean(overlap_rate))
    imp_mean_enet.append(np.mean(imp_rate))
    rmse_mean_enet.append (np.mean(rmse)) 
    end_time = time.monotonic()
    delta_t = timedelta(seconds=end_time - start_time)
    #segr_ridge =(cvResult(overlap_rate_mean_ridge, coefdict_ridge, i, 0.9 ))
    #Test_seg = BottomList(endal_seg.iloc[test_indice], 'base_seg_id', 0.05)
    #intersect = len(Intersection(segr_ridge.keys(), Test_seg))/len(Test_seg)
    print (ii, delta_t, np.mean(imp_rate),np.mean(rmse),np.mean(overlap_rate) )
    ii +=1


# In[159]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth = 4,
                                min_samples_leaf = 0.1,
                                random_state = 3)


# In[160]:


tree_y_pred = tree_reg.fit(X_train, y_train).predict(X_test)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_y_pred))
tree_rmse


# In[161]:


importances_dr = pd.Series(tree_reg.feature_importances_, index = X_train.columns)
sorted_importance_dr = importances_dr.sort_values(ascending = False)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor 

# Instantiate rf
rf = RandomForestRegressor(n_estimators=4,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 


# In[ ]:


rf_y_pred = rf.predict (X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_rmse


# In[56]:


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
AND st.advertiser_name = 'Cinemax - US'
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


# In[57]:


#endal_10


# In[58]:


endal_10 = endal_test.copy()
endal_10['conversions'].fillna(0, inplace = True)
endal_10['cr'] = endal_10['conversions']/endal_10['impressions']

print(endal_10['cr'].mean(), endal_10['cr'].std())


# In[59]:


# split data by ad id
gb = endal_10.groupby('ad_cfid')    
df_camp = [gb.get_group(x) for x in gb.groups]


# In[590]:


suggestseg = seg_id_n


# In[60]:


suggestseg1  = seid
len(suggestseg1)


# In[61]:


len(suggestseg1)


# In[62]:


# suggestseg2 = seg_cand2[0:307]
# len(suggestseg2)


# In[63]:


#len(list(set(suggestseg1) & set(suggestseg2)))


# In[729]:


#suggestseg3 = list(set(suggestseg1) & set(suggestseg2))


# In[64]:


test_lsn = len(df_camp)
overlap_rep_ridge = []
se = []
itp = []
imp = []
decrease_impression = []
est_seg_ridge = suggestseg1
for i in range (test_lsn):
    df_test = df_camp[i]
    segn = len(df_test['base_seg_id'].unique())
    if segn>1000:
        Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
        ovlr = len(Intersection(est_seg_ridge, Test_seg_test))/len(Test_seg_test)
        se.append(segn)
        decr = impression_de(df_test,est_seg_ridge )
        overlap_rep_ridge.append(ovlr)
        if df_test['cr'].mean()!=0:
            impr = improve_f(df_test,est_seg_ridge)
            impr_it = improve_f(df_test,Test_seg_test)
            imp.append(impr)
            itp.append(impr_it)
            decrease_impression.append(decr)


# In[65]:


len(imp)


# In[68]:


imp)


# In[69]:


np.mean(itp)


# In[70]:


np.mean(itp)


# In[71]:


np.min(decrease_impression)


# In[808]:


def adTest (df, seg):
    gb = df.groupby('ad_cfid')    
    df_camp = [gb.get_group(x) for x in gb.groups]
    test_lsn = len(df_camp)
    overlap_rep_ridge = []
    se = []
    itp = []
    imp = []
    est_seg_ridge = seg
    for i in range (test_lsn):
        df_test = df_camp[i]
        segn = len(df_test['base_seg_id'].unique())
        if segn>1000:
            Test_seg_test = BottomList(df_test, 'base_seg_id', 0.1)
            ovlr = len(Intersection(est_seg_ridge, Test_seg_test))/len(Test_seg_test)
            se.append(segn)
            decr = impression_de(df_test,est_seg_ridge )
            overlap_rep_ridge.append(ovlr)
            if df_test['cr'].mean()!=0:
                impr = improve_f(df_test,est_seg_ridge)
                impr_it = improve_f(df_test,Test_seg_test)
                imp.append(impr)
                itp.append(impr_it)
                decrease_impression.append(decr)
    return np.mean(imp)
        
    
    


# In[789]:


np.min(decrease_impression)


# In[586]:


np.mean(imp)


# In[786]:


overlap_rep_ridge


# In[572]:


adTest (endal_10,seg_id_n )


# In[ ]:




