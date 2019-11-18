import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from pathlib import Path
import itertools
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data')

onetime = pd.read_csv(data_folder/"mosmonetime_EA.csv")
onetime.columns
onetime['MOsID'] = onetime['Type'].str.extract("(\d*\.?\d+)", expand=True)
onetime['MOs'] = onetime['Type'].str.extract('([A-Z]\w{0,})', expand=True)
onetime.columns
WCMO = onetime[onetime['MOs'] == "WCMO"]
wcmo_sortby = WCMO.sort_values(by=['sedbcr'], ascending=False).reset_index()
wcmo_sortby['sedbcr_dist'] = abs(wcmo_sortby['sedbcr'].diff( periods = 1)) 

wcmo_sortby.to_csv(data_folder/"wcmo_sortby.csv", index = False)
onetime['sedbcr']
onetime_sortby = onetime.sort_values(by=['sedbcr'], ascending=False).reset_index()
onetime_sortby['sedbcr_dist'] = abs(onetime_sortby['sedbcr'].diff( periods = 1)) 
onetime_sortby.to_csv(data_folder/"onetime_sortby.csv", index = False)
onetime_sortby.columns

MOSMpointsSB574 = pd.read_csv(data_folder/"MOSMpointsSB574.csv")
mosmdata = pd.read_csv(data_folder/"mosmdata.csv")
MOSMpointsSB574.MOs.describe()
mosmdata.columns
mosmdata.dtypes
MOSMpointsSB574.dtypes


new_df = mosmdata.merge(MOSMpointsSB574, left_on=['MOs', 'MOsID'], right_on = ['MOs', 'MOsID'], how='left')
new_df.columns
new_df = new_df.fillna(0)
new_df.to_csv(data_folder/"onetime_EA_subbasins.csv", index = False)
###################################################################################


## Creat seeds:
mosmfile16891 = pd.read_csv(data_folder/"onetime_EA_subbasins.csv")
NBR_ITEMS = mosmfile16891.shape[0]
mosmfile16891.columns
pmt = itertools.product([0,1], repeat=3)
pmtlist = list(pmt)
subs = np.array([19,21,23])
subs_exp = pmtlist*subs
subs_exp

sce = np.arange(1,9,1)
sce
sce_name = ["80000" + str(i) for i in sce]
sce_name

my_list = subs_exp[3][np.nonzero(subs_exp[3])[0]]

practices = ["WCMO", "ICMO"]
mosmfile16891['Subbasin'].isin(my_list) 
mosmfile16891['MOs'].isin(practices) 

mosmfile16891[(mosmfile16891['MOs'].isin(practices)) & (mosmfile16891['Subbasin'].isin(my_list))]

for t in range(len(sce)):
    conditions = subs_exp[t][np.nonzero(subs_exp[t])[0]]
    mosmfile16891[sce_name[t]] = [1]* NBR_ITEMS
    mosmfile16891.loc[mosmfile16891[(mosmfile16891['MOs'].isin(practices)) & (mosmfile16891['Subbasin'].isin(conditions))].index,sce_name[t]] = 4
  
        
  
mosmfile16891['800001']

mosmfile16891.to_csv(data_folder/"EXPseeds.csv", index = False)
#####################

seedsfile = pd.read_csv(data_folder/"EXPseeds.csv")
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
df_a.columns = onlyseeds.columns

pd.concat([onlyseeds, df_a])


mosmfile16891[sce_name[0]].to_csv('seed1example.txt', sep='\t')
np.savetxt(data_folder/'testseeds.txt', mosmfile16891[sce_name[0]], delimiter='')  
