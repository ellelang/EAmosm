import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from pathlib import Path
import itertools
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data')
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame


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






sub574 = gpd.read_file(data_folder/"shapefiles/SB574.dbf")
sub574_Table = pd.DataFrame(sub574)
sub574_Table.columns
sub574_Table.geometry

sub574_bcrdis = sub574_Table.merge(ave_dis_df, left_on="SB574", right_on="SB574")
sub574_bcrdis.geometry

sub574_bcrdis_bdf = GeoDataFrame(sub574_bcrdis, geometry = sub574_bcrdis.geometry)
sub574_bcrdis_bdf.to_file(data_folder/"shapefiles/SB574_bcrdist.shp")



##################mosmfile16891.columns
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

