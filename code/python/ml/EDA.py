# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('bmh')
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data/forML/original')
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report

df = pd.read_csv(data_folder/'mosmdata.csv')

df.columns
data_m = df.loc[:, 'GeneID':'coords_x2']
data_m.columns
data_m['Cost'].replace(0, np.nan, inplace=True)
# Replace cost=0 by the meidan
data_m['Cost'] = data_m.groupby(['MOs'])['Cost']\
    .transform(lambda x: x.fillna(x.median()))
data_m['sedbcr'] = data_m['SedRed']*1000 /data_m['Cost']   
data_m['duckbcr'] = data_m['Duck'] /data_m['Cost']           
data_m.isnull().sum()

data_m2 = data_m[['GeneID', 'MOsID', 'MOs', 'SedRed','Duck', 'Cost', 'sedbcr', 'duckbcr', 'SB574', 'HydroSB', 'Zone',
       'watershed', 'coords_x1', 'coords_x2']]


ld = np.round(np.arange(0,1.1,0.1),1)
ld
ldname = ["ld" + str(i) for i in ld]
ldname

def ldfun (x, y, lda):
    return (x * lda + y * (1-lda)) 
    
for i in range(len(ld)):
   data_m2[ldname[i]] = ldfun (data_m2.sedbcr,data_m2.duckbcr, ld[i])

data_m2.columns
data_m2.to_csv(data_folder/'mosmdata_m.csv', index = False, na_rep='NA')


data_wcmo = data_m2[data_m2['MOs']=='WCMO']
data_tlmo = data_m2[data_m2['MOs']=='TLMO']
data_afmo = data_m2[data_m2['MOs']=='AFMO']
data_bfmo = data_m2[data_m2['MOs']=='BFMO']
data_ncmo = data_m2[data_m2['MOs']=='NCMO']
data_icmo = data_m2[data_m2['MOs']=='ICMO']
data_ramo = data_m2[data_m2['MOs']=='RAMO']

data_wcmo.info()
data_tlmo.info()
data_afmo.info()
data_bfmo.info()
data_ncmo.info()
data_icmo.info()
data_ramo.info()

dfwcmo = pd.read_csv(data_folder/'wcmo.csv')
dfwcmo.info()
data_wcmo.info()
dftlmo = pd.read_csv(data_folder/'tlmo.csv')
dftlmo.info()
dftafo = pd.read_csv(data_folder/'afmo.csv')
dftafo.info()
dfbfmo = pd.read_csv(data_folder/'bfmo.csv')
dfbfmo.info()
dfncmo = pd.read_csv(data_folder/'ncmo.csv')
dfncmo.info()
dficmo = pd.read_csv(data_folder/'icmo.csv')
dficmo.info()
dframo = pd.read_csv(data_folder/'ramo.csv')
dframo.info()

#######################

wcmo_merge = data_wcmo.merge(dfwcmo, left_on='MOsID', right_on='Site_ID')

wcmo_merge.head(6)

mosms = ['AFMO', 'BFMO', 'ICMO','NVMO', 'RAMO','TLMO', 'WCMO']

# Iterate through the five airlines
for m in mosms:
    # Subset to the airline
    subset = data_m[data_m['MOs'] == m]
    subset['Cost'].replace(0,subset['Cost'].median())
    # Draw the density plot
    sns.distplot(subset['Cost'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = m)

cost_groupmean = data_m.groupby('MOs')['Cost'].mean().reset_index()
cost_groupmedian = data_m.groupby('MOs')['Cost'].median().reset_index()



# Replace cost=0 by the meidan
data_m['Cost'] = df.groupby(['MOs'])['Cost']\
    .transform(lambda x: x.fillna(x.median()))

data_m.to_csv(data_folder/'mosmdata_m.csv')

data_m[data_m['MOs'] =='AFMO']['Cost']
data_m.to_csv(data_folder/'mosmdata_m.csv', index = False, na_rep='NA')