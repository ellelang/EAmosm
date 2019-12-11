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
newdf = mosmfile16891.sort_values(by=['SED_BCR_x'],ascending=False).reset_index()
newdf.shape
newdf['sedbcr_dist'] = abs(newdf['SED_BCR_x'].diff( periods = 1))
newdf['sedbcr_dist'][0] = 0
#newdf_nozero = newdf.loc[newdf['SED_BCR_x'] != 0]
#newdf_nozero.shape
#newdf_nozero['sedbcr_dist'] = abs(newdf_nozero['SED_BCR_x'].diff( periods = 1)) 
#newdf_nozero.columns
#np.min(newdf_nozero['SB574'])
ave_dis = newdf.groupby(['SB574'])['sedbcr_dist'].mean()
ave_dis_df = ave_dis.add_suffix(' ').reset_index()
ave_dis_df['SB574'] = ave_dis_df['SB574'].astype(int)

sub574 = gpd.read_file(data_folder/"shapefiles/SB574.shp")
sub574_Table = pd.DataFrame(sub574)
sub574_Table.columns
sub574_Table.geometry

sub574_bcrdis = sub574_Table.merge(ave_dis_df, left_on="SB574", right_on="SB574")
sub574_bcrdis.geometry

sub574_bcrdis_bdf = GeoDataFrame(sub574_bcrdis, geometry = sub574_bcrdis.geometry)
sub574_bcrdis_bdf.to_file(data_folder/"shapefiles/SB574_bcrdist.shp")

##################mosmfile16891.columns ld 0, 0.5, 1 #### 
ld = np.arange (0, 1.1, 0.1)
ldnames = ["lambdased" + f'{i:.1f}' for i in ld]
for t in range(len(ld)):
    mosmfile16891[ldnames[t]] = mosmfile16891['SED_BCR_x'] * ld[t] + \
    mosmfile16891['Duck_BCR_x'] * (1 - ld[t]) 

ld_distsce = ["dist_ld" + f'{i:.1f}' for i in ld]
newdf = mosmfile16891


newdf['lambdased1.0']
newdfld1 = newdf.sort_values(by=['lambdased1.0'], ascending=False).reset_index()
newdfld1['sedbcr_dist'] = abs(newdf['SED_BCR_x'].diff( periods = 1))                                    
######################################
#from pyproj import Proj


sub_ld574 = gpd.read_file (data_folder/"shapefiles/dist_sub574.shp")
sub_30 = gpd.read_file (data_folder/"shapefiles/subbasins.shp")
streams = gpd.read_file (data_folder/"shapefiles/LeSueur_Streams.shp") 
sub_ld574.columns
sub_30.columns
sub_ld574['disld1'] = sub_ld574['disld1'].fillna(0)
sub_ld574['disld05'] = sub_ld574['disld05'].fillna(0)
sub_ld574['disld0'] = sub_ld574['disld0'].fillna(0)* 10000

#sub_ld574.to_crs(epsg = 3857)

plt.rcParams["legend.fontsize"] = 12

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(45, 15))
sub_ld574.plot(ax = ax[0], column = 'disld1', scheme = 'natural_breaks', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True)      
sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[0], color = 'blue', legend = False)
ax[0].set_axis_off() 
ax[0].title.set_text(r'$\lambda = 1$')  
ax[0].title.set_fontsize(25)
         
sub_ld574.plot(ax = ax[1], column = 'disld05', scheme = 'natural_breaks', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True)      
sub_30.plot(ax = ax[1], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[1], color = 'blue', legend = False)
ax[1].set_axis_off() 
ax[1].set_axis_off() 
ax[1].title.set_text(r'$\lambda = 0.5$')  
ax[1].title.set_fontsize(25) 

sub_ld574.plot(ax = ax[2], column = 'disld0', scheme = 'natural_breaks', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True)      
sub_30.plot(ax = ax[2], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[2], color = 'blue', legend = False)
ax[2].set_axis_off()  
ax[2].set_axis_off() 
ax[2].title.set_text(r'$\lambda = 0$')  
ax[2].title.set_fontsize(25)     
plt.savefig(data_folder/'choropleth_bcrdist.pdf', bbox_inches='tight', dpi = 600)