#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:26:24 2022

@author: ellelang
"""

from pathlib import Path
data_folder =  Path('/Users/ellelang/Desktop/github/EAmosm/data/MRB2022')


import pandas as pd 
import numpy as np
from pprint import pprint 
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

dat1aat = pd.read_csv (data_folder/"1AAT1014compile.csv")
## Some RAMO HAS COST ==0 
dat1aat = dat1aat.query('Cost != 0')
genome_code = pd.read_csv (data_folder/"genomecode.csv")
genome_code.head()
dat1_join = pd.merge(dat1aat, genome_code, how = 'left', on = 'AlleleCombo')
dat1_join.head()
dat1_join.columns
nsize = len (dat1_join.index)
nsize

obid = dat1_join ['ID']
oblabel = dat1_join ['Label']
Ob_sed = dat1_join ['SedRed'] 
Ob_nit = dat1_join ['NitRed']
Ob_cost = dat1_join ['Cost']
ob_genomesection = dat1_join ['Genome section']
ob_wintin = dat1_join['WithinSectionID']
ob_value = dat1_join['Value']
ob_description = dat1_join['Describe']

dat1_join

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
    'Genome':ob_genomesection,
    'id_within':ob_wintin,
    'value': ob_value,
    'MagOpt': ob_description
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
df_dataset.head()
df_dataset.to_csv(data_folder/'lambda_1aat1014.csv', index = False)
#####################3Rank within the Genome
df_dataset.Genome.unique()
df_dataset_subISO = df_dataset[df_dataset.Genome == 'subISO']
ldrankname = ["rankbyld" + str(i) for i in ld]
ldrankname
for i in range (len(ld)):
    df_dataset[ldrankname[i]] = df_dataset.groupby(['Genome', 'id_within'])[ldname[i]].rank("first", ascending =False)

df_dataset.head()
df_dataset.to_csv(data_folder/'withinrank_1aat1014.csv', index = False)
genomelist = ['subISO', 'subInC', 'HRU', 'RAMO']
#genomelist = ['subISO']
df_dataset.head()
df_dataset.columns 
ld_scenario_lst = []
for j in range (len(ld)):
    select_df_lst = []
    for i in genomelist:
        
        df_geno_i = df_dataset[(df_dataset['Genome'] == i) & (df_dataset[ldrankname[j]] == 1)]
        columns_report = ['ID', 'Label', 'SRed', 'NRed', 'Cost', 'BCR_SC', 'BCR_NC', 'Genome',
           'id_within', 'value', 'MagOpt', ldname[j]]
        select_df_lst.append(df_geno_i[columns_report])
    
    df_geno_total = pd.concat(select_df_lst, axis = 0)
    ld_scenario_lst.append(df_geno_total)  
    
len(ld_scenario_lst)
ld_scenario_lst[0].index
#select_df_lst = []
ld_scenario_lst[0].head()
ld_scenario_lst[0]['ld0.0'].rank(ascending =False)
top = np.round(np.arange(0.1,1,0.1),1)
top

top_pct = top*100
topname = ["top" + str(int(i)) for i in top_pct]
topname
ldtopname = [x + y for x in ldname for y in topname]
#len(ldtopname)
# ldtopname
# nameindex = list (range(0,99,9))
# nameindex


def rankAcross (df_ld, ld_column): 
    nsize = len(df_ld.index)
    
    df_ld['rankby'] = df_ld[ld_column].rank(ascending =False)
    
    for t in range (len(top)):
        topby = int(round(top[t] * nsize ))
        seed_ld = [0]*df_ld.shape[0] 
        
        for j in range (nsize) :
            
            if df_ld['rankby'].tolist()[j] <= topby:
                #print(df_ld['rankby'].tolist()[j])
                seed_ld [j] = df_ld['value'].tolist()[j]
            else:
                seed_ld [j] = 1
            #print (seed_ld[j])
        seeds_name = ld_column + topname[t]
        df_ld[seeds_name] = seed_ld
    
    return df_ld

trydf = rankAcross(ld_scenario_lst[0], 'ld0.0')
trydf[trydf.columns[trydf.columns.isin(ldtopname)]] 

trydf.isin(ldtopname)

seeds_df = []

for i in range(len(ldname)):
    df = ld_scenario_lst[i]
    col = ldname[i]
    rep = rankAcross(df, col)
    #seeds_ld = rep[rep.columns[rep.columns.isin(ldtopname)]]
    
    seeds_df.append(rep)
seeds_df[0].shape  

with pd.ExcelWriter(data_folder/'seeds_acrossgenome1014.xlsx') as writer:
    for i in range(len(ldname)):
        report = seeds_df[i]
        report.to_excel(writer, sheet_name= ldname[i])

onlyseeds_lst = []
for i in range(len(seeds_df)):
    only_seeds_i = seeds_df[i].iloc[:,-9:]
    only_seeds_i.reset_index(drop=True, inplace=True)
    onlyseeds_lst.append(only_seeds_i)

onlyseeds = pd.concat(onlyseeds_lst, axis = 1)
onlyseeds.head()


#random_end = np.array([-1 , 27732.9131,  149929377.3994,  49015.6441,  897.6627,  1021.1096,  211.799,  4626,  0,  2119,  517,  7874,  537,  106,  1112 , 546, 0, 30969626.3368])

random_end = np.array([-1 ,2783558989.0477, 189250.710042, 20219.2666, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, \
                       0.0000, 0.0000, 0.0000, 0.0000, 512.0000 ,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\
                    266.199692, 163851260.047695, 109379.4818, 188871.3991, 568806.8462, 20219.2585, 1447821.0951, 1023660.701613,\
                        451434.367985, 930763.807542, 410466.846548, 266.199692, 163851260.047695, 109399.5216, 507517.028962, 1447821.0951])

random_end.shape



a = np.tile(random_end,(len(ldtopname),1))
a.shape
a_trans = np.transpose(a)

df_a = pd.DataFrame(a_trans, index=range(a_trans.shape[0]), columns=range(a_trans.shape[1]))
df_a.columns = onlyseeds.columns
df_a.shape

seeds_ALL = pd.concat([onlyseeds, df_a]).reset_index().drop(['index'],axis=1)
seeds_ALL.columns
seeds_ALL.shape

len(top)
len(ld)

sce_name1 = [str(int(i*10))  for i in ld]
sce_name2 = [str(int(i*100)) for i in top]

sce_name2

import itertools
mylist = list(itertools.product(sce_name1, sce_name2))
namelist = [ '1000' + s1 + s2  for s1, s2 in mylist]
len(namelist)
namelist

seedstxt = []

    #seeds_df[c].to_csv(c + '.txt', index=False)
for c in seeds_ALL.columns:
    seeds_sce = seeds_ALL[c].tolist()
    col_index = seeds_ALL.columns.get_loc(c)
    seeds_sce.insert(0, int(namelist[col_index]))
    seedstxt.append(seeds_sce)
    
aaa = np.array(seedstxt)
np.savetxt(data_folder/'MRBSeeds20221014.txt', aaa,  delimiter=' ', fmt='%d')    









ldname
for i in genomelist:
    print(i)
    df_geno_i = df_dataset[(df_dataset['Genome'] == i) & (df_dataset['rankbyld0.8'] == 1)]
    columns_report = ['ID', 'Label', 'SRed', 'NRed', 'Cost', 'BCR_SC', 'BCR_NC', 'Genome',
       'id_within', 'value', 'ld0.8']
    select_df_lst.append(df_geno_i[columns_report])

df_geno_total = pd.concat(select_df_lst, axis = 0)
df_geno_total   
select_df_lst[0]
    
df_dataset_subISO.shape
3090/6
df_dataset_subISO.columns
df_dataset_subISO.id_within

df_dataset_subISO["rank0.6"] = df_dataset_subISO.groupby("id_within")["ld0.6"].rank("dense", ascending=False)



grouped = df_dataset_subISO.groupby('id_within').sort_value(by = 'ld0.6' , ascending =False)

value_lst = []
for i in range (len (ld)):
    
    rankby = df_dataset[ldrankname[i]]


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











