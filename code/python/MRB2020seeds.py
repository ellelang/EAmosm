from pathlib import Path
#data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
data_folder = Path('/Users/ellelang/Documents/github/EAmosm/data')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
import random
import json
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

dat = pd.read_csv (data_folder/'MRB2020/MRB1AAT1016.csv')
dat = dat.iloc[1:]
dat.columns

dat['NitRed'] = np.where(dat['NitRed']==0, 
                       np.random.uniform(1, 100, size=len(dat)), 
                       dat['NitRed'])

dat['SedRed'] = np.where(dat['SedRed']==0, 
                       np.random.uniform(1, 100, size=len(dat)), 
                       dat['SedRed'])

demodata = dat[(dat['SedRed']!= 0) & (dat['NitRed'] != 0)]
demodata.head(2)
nsize = len (demodata.index)
nsize


obid = demodata ['ID']
oblabel = demodata ['Label']
Ob_sed = demodata ['SedRed'] 
Ob_nit = demodata ['NitRed']
Ob_cost = demodata ['Cost']


bcr_sc = [x/y for x, y in zip(Ob_sed , Ob_cost)]
bcr_nc = [x/y for x, y in zip(Ob_nit, Ob_cost)]


dict_new = {
    'ID': obid,
    'Label': oblabel,
    'SRed': Ob_sed,
    'NRed':Ob_nit,
    'Cost':Ob_cost,
    'BCR_SC': bcr_sc,
    'BCR_NC': bcr_nc,
}

ks = pd.DataFrame(dict_new)
ld = np.round(np.arange(0,1.1,0.1),1)
ldname = ["ld" + str(i) for i in ld]
ldname
wht_ld = [[0] * nsize for j in range(len(ld))]
for j in range(len(ld)):
    ld_value = ld [j]
    wht_ld[j] = [ld_value * x + (1-ld_value) * y for x, y in zip(bcr_sc  , bcr_nc)]


ld_array = np.transpose(np.array(wht_ld))
ld_array
ld_df = pd.DataFrame(ld_array)
ld_df.columns = ldname

df_dataset = pd.concat([ks.reset_index(drop=True), ld_df], axis=1)
ldrankname = ["rankbyld" + str(i) for i in ld]
ldrankname
for i in range (len(ld)):
    df_dataset[ldrankname[i]] = df_dataset[ldname[i]].rank(ascending =False)
    
top = np.round(np.arange(0.1,1,0.1),1)
top

top_pct = top*100
topname = ["top" + str(int(i)) for i in top_pct]
topname
ldtopname = [x + y for x in ldname for y in topname]
len(ldtopname)
ldtopname
nameindex = list (range(0,99,9))
nameindex

for i in range (len (ld)):
    rankby = df_dataset[ldrankname[i]]
    for t in range (len (top)):
        topby = int(round(top[t] * nsize ))
        seed_ld = [0]*df_dataset.shape[0] 
        for j in range (nsize) :
            if rankby [j] <= topby:
                seed_ld [j] = 1
            else:
                seed_ld [j] = 0
        df_dataset[ldtopname[nameindex[i]+t]] = seed_ld
        
        
df_dataset.to_csv(data_folder/"MRB2020/demodata_seeds.csv",index = False, sep=',', encoding='utf-8')
df_dataset.columns
df_dataset.head(4)

###############Objective space
nit_base = 26257132.9431
sed_base = 590389.1611

df_ld = pd.concat([ks.reset_index(drop=True), ld_df], axis=1)
df_ld.columns

df_ld.sort_values(by = 'ld0.0', inplace = True, ascending = False ) 
df_ld.head(4)
df_ld[['SRed', 'NRed','Cost']]

def ObjectCumSum (df,col_name):
    df_sorted= df.sort_values(by = col_name, inplace = True, ascending = False )
    df_obj = df[['SRed', 'NRed','Cost']]
    df_obj_sum = df_obj.cumsum(axis = 0)
    df_obj_sum['ld'] = col_name
    return df_obj_sum

#df_09 = ObjectCumSum(df_ld,'ld0.9')
#df_09.head(4)



appended_data = []

for i in ldname:
    df_cumsum = ObjectCumSum(df_ld, i)
    appended_data.append(df_cumsum)
    

df_obj = pd.concat(appended_data)
df_obj.shape

df_obj['origin'] = 'wBCR'

sample_range = range (0, len(df_obj), 200)
len(list(sample_range))

#df_obj1 = df_obj.iloc[sample_range]\
df_obj1 = df_obj[df_obj['ld'].isin (['ld0.0','ld0.1','ld0.2',
                                     'ld0.3','ld0.4','ld0.5',
                                     'ld0.6','ld0.7','ld0.8',
                                     'ld0.9'])]

ax = plt.axes(projection='3d')
ax.scatter3D(df_obj.NRed, df_obj.SRed, df_obj.Cost, \
             color = '#17202A', label = "wBCR", marker = 'o',
             linewidth=0.001)

ax.set_xlabel('Sediment Reduction',labelpad=5)
ax.set_ylabel('NO3 Reduction',labelpad=5)
ax.set_zlabel('Cost ($/Year)',labelpad=5)
ax.legend(loc=2, fontsize = 5)
plt.show() 

df_obj['ld'].unique()
NO3Cost = sns.lmplot(x='NRed', y='Cost', hue='ld', data=df_obj, scatter_kws={"s": 30, 'alpha': 0.8},
                     fit_reg=False) 

SedCost = sns.lmplot(x='SRed', y='Cost', hue='ld', data=df_obj, scatter_kws={"s": 30, 'alpha': 0.8},
                     fit_reg=False)   
    
###########Create seeds
df_dataset.columns
#onlyseeds = df_dataset.iloc[:,-len(ldtopname):]
#onlyseeds.columns

# selectseeds_ld = onlyseeds.loc[:, ["ld0.0top10","ld0.0top30","ld0.0top60",
#                                     "ld0.5top10","ld0.5top30","ld0.5top60",
#                                    "ld1.0top10","ld1.0top30","ld1.0top60" ]]

selectseeds_ld = df_dataset.loc[:, ["Label","ld0.0top10","ld0.0top30","ld0.0top60",
                                    "ld0.5top10","ld0.5top30","ld0.5top60",
                                   "ld1.0top10","ld1.0top30","ld1.0top60" ]]

selectseeds_ld.to_csv(data_folder/'MRB2020/selectld.csv', index = False)





random_end = np.array([-1 , 27732.9131,  149929377.3994,  49015.6441,  897.6627,  1021.1096,  211.799,  4626,  0,  2119,  517,  7874,  537,  106,  1112 , 546, 0, 30969626.3368])
random_end.shape


a = np.tile(random_end,(len(ldtopname),1))
a.shape
a_trans = np.transpose(a)

df_a = pd.DataFrame(a_trans, index=range(a_trans.shape[0]), columns=range(a_trans.shape[1]))
df_a.columns = onlyseeds.columns
df_a.head(2)
seeds_df = pd.concat([onlyseeds, df_a]).reset_index().drop(['index'],axis=1)
seeds_df.columns
seeds_df.head



len(top)
len(ld)

sce_name1 = [str(int(i*10))  for i in ld]
sce_name2 = [str(int(i*10)) for i in top]

import itertools
mylist = list(itertools.product(sce_name1, sce_name2))
namelist = [s1 + s2 + '800000' for s1, s2 in mylist]
namelist

#seeds_df.columns
seedslist = []

    #seeds_df[c].to_csv(c + '.txt', index=False)
for c in seeds_df.columns:
    seeds_sce = seeds_df[c].tolist()
    col_index = seeds_df.columns.get_loc(c)
    seeds_sce.insert(0, int(namelist[col_index]))
    seedslist.append(seeds_sce)
    
aaa = np.array(seedslist)
aaa
np.savetxt(data_folder/'MRB2020/ExperimentSeeds.txt', aaa,  delimiter=' ', fmt='%d')    


