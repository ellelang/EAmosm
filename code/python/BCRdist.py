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

pmt = itertools.product([0,1], repeat=3)
pmtmatrix = np.matrix(list(pmt))
sub_val = np.array([19,21,23])
