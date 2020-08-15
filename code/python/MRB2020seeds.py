from pathlib import Path
#data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
data_folder = Path('/Users/zhenlang/Desktop/github/EAmosm/data')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
import random
import json

dat = pd.read_csv (data_folder/'MRB2020/MRB2020seedTest.csv')
demodata = dat[(dat['SedRed']!= 0) & (dat['NitRed'] != 0)]
demodata.head(2)
nsize = len (demodata.index)
nsize


obid = demodata ['ID']
Ob_sed = demodata ['SedRed'] 
Ob_nit = demodata ['NitRed']
Ob_cost = demodata ['Cost']


bcr_sc = [x/y for x, y in zip(Ob_sed , Ob_cost)]
bcr_nc = [x/y for x, y in zip(Ob_nit, Ob_cost)]



dict_new = {
    'ID': obid,
    'SRed': Ob_sed,
    'NRed':Ob_nit,
    'Cost':Ob_cost,
    'BCR_SC': bcr_sc,
    'BCR_NC': bcr_nc,
}

ks = pd.DataFrame(dict_new)
ld = np.round(np.arange(0,1.1,0.1),1)
ldname = ["ld" + str(i) for i in ld]

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
        
        
#df_dataset.to_csv(data_folder/"MRB2020/demodata_seeds.csv",index = False, sep=',', encoding='utf-8')
df_dataset.columns


###########Create seeds


onlyseeds = df_dataset.iloc[:,-len(ldtopname):]
onlyseeds.columns

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