import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from pathlib import Path
import itertools
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data')
mosmfile16891 = pd.read_csv(data_folder/"onetime_EA_subbasins.csv")
wcmoga = pd.read_csv(data_folder/"wcmogawhole.csv")
icmoga = pd.read_csv(data_folder/"icmogawhole.csv")
wcmo_subset = wcmoga[ (wcmoga['HYDSB_LES30SB']==28) | (wcmoga['HYDSB_LES30SB']==25 )| (wcmoga['HYDSB_LES30SB']==19)]
icmo_subset = icmoga[ (icmoga['Subbasin_2']==28) | (icmoga['Subbasin_2']==25 )| (icmoga['Subbasin_2']==19)]
wcmoSedRed = wcmo_subset.groupby(['HYDSB_LES30SB'])['SedRed'].sum().reset_index()
icmoSedRed = icmo_subset.groupby(['Subbasin_2'])['SedRed'].sum().reset_index()

wcmoSedRed.columns = ['Subbasin','SedRed']
icmoSedRed.columns = ['Subbasin','SedRed']
#SedRedSum = pd.merge([wcmoSedRed, icmoSedRed])
SedSum = pd.concat([wcmoSedRed,icmoSedRed]).groupby(['Subbasin'])['SedRed'].sum().reset_index()
SedSum

pmt = itertools.product([0,1], repeat=3)
pmtmatrix = np.matrix(list(pmt))
sumSed = pmtmatrix.dot(SedSum['SedRed'])

wcmo_subset.groupby(['HYDSB_LES30SB']).count()
icmo_subset.groupby(['Subbasin_2']).count()
#############3
subs = np.array([19,25,28])
practices = ["WCMO", "ICMO"]
mosmfile16891['Subbasin'].isin(subs) 
mosmfile16891['MOs'].isin(practices) 

mosmsubset_192528 = mosmfile16891.loc[mosmfile16891[(mosmfile16891['MOs'].isin(practices)) & (mosmfile16891['Subbasin'].isin(subs))].index,]
SedSum_check = mosmsubset_192528.groupby(['Subbasin'])['SED_B'].sum().reset_index()
SedSum_check
pmt = itertools.product([0,1], repeat=3)
pmtmatrix = np.matrix(list(pmt))
sumSed_check = pmtmatrix.dot(SedSum_check['SED_B'])


## Creat seeds:
mosmfile16891 = pd.read_csv(data_folder/"onetime_EA_subbasins.csv")
NBR_ITEMS = mosmfile16891.shape[0]
mosmfile16891.columns
pmt = itertools.product([0,1], repeat=3)
pmtlist = list(pmt)
subs = np.array([19,25,28])

subs_exp = pmtlist*subs
subs_exp

sce = np.arange(1,9,1)
sce
sce_name = ["80000" + str(i) for i in sce]
sce_name

my_list = subs_exp[3][np.nonzero(subs_exp[3])[0]]
mosmfile16891[(mosmfile16891['MOs'].isin(practices)) & (mosmfile16891['Subbasin'].isin(my_list))]

for t in range(len(sce)):
    conditions = subs_exp[t][np.nonzero(subs_exp[t])[0]]
    mosmfile16891[sce_name[t]] = [1]* NBR_ITEMS
    mosmfile16891.loc[mosmfile16891[(mosmfile16891['MOs'].isin(practices)) & (mosmfile16891['Subbasin'].isin(conditions))].index,sce_name[t]] = 4
  
        
  
mosmfile16891['800001']

mosmfile16891.to_csv(data_folder/"EXPseeds1118.csv", index = False)
##################### seeds 

seedsfile = pd.read_csv(data_folder/"EXPseeds1118.csv")
seedsfile.shape
seedsfile.columns

onlyseeds = seedsfile.iloc[:,-8:]
onlyseeds.shape
onlyseeds.columns
random_end = np.array([-1 , 27732.9131,  149929377.3994,  49015.6441,  897.6627,  1021.1096,  211.799,  4626,  0,  2119,  517,  7874,  537,  106,  1112 , 546, 0, 30969626.3368])
random_end.shape

a = np.tile(random_end,(8,1))
a.shape
a_trans = np.transpose(a)

df_a = pd.DataFrame(a_trans, index=range(a_trans.shape[0]), columns=range(a_trans.shape[1]))
df_a.columns
df_a.columns = onlyseeds.columns

seeds_df = pd.concat([onlyseeds, df_a]).reset_index().drop(['index'],axis=1)
seeds_df.columns
seeds_df.head

seedslist = []
for c in seeds_df.columns:
    seeds_sce = seeds_df[c].tolist()
    seeds_sce.insert(0, int(c))
    seedslist.append(seeds_sce)
    #seeds_df[c].to_csv(c + '.txt', index=False)

aaa = np.array(seedslist)
np.savetxt(data_folder/'ExperimentSeeds1118.txt', aaa,  delimiter=' ', fmt='%d')    

mosmfile16891[sce_name[0]].to_csv('seed1example.txt', sep='\t')
np.savetxt(data_folder/'testseeds.txt', mosmfile16891[sce_name[0]], delimiter='')  