from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
#data_folder = Path('/Users/zhenlang/Desktop/github/EAmosm/data')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import array
import random
import json
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import matplotlib.patches as mpatches
#import pyepsg


selectseeds = pd.read_csv(data_folder/'MRB2020/selectld.csv')
selectseeds.head(3)
sub_id = pd.read_csv(data_folder/"MRB2020/subbasin_ids.csv")
sub_id.head(3)

sub_seedsmerge = pd.merge(sub_id, selectseeds, how = 'left', left_on = 'Label', right_on = 'Label') 
sub_seedsmerge

practice = sub_seedsmerge.Practices.unique().tolist()
practice

col_ld = ["ld0.0top10","ld0.0top30","ld0.0top60",\
"ld0.5top10","ld0.5top30","ld0.5top60",\
"ld1.0top10","ld1.0top30","ld1.0top60" ]   
    
######Specify the columns
sub_new = sub_seedsmerge.groupby("subbasin")["ld0.0top10"].apply(list).reset_index(name='new')
sub_new.columns
sub_new.head(3)

sub_result = pd.DataFrame(sub_new['new'].to_list(), columns=practice)
sub_result['subbasin'] = sub_new['subbasin'].str.extract('(\d+)').astype(int)
sub_result.head(3)
sub_result.to_csv(data_folder/"MRB2020/selected_spatial/subbasin_ld0top10.csv", index = False)





#######################
subbasin = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_5_subbasins.shp")
subbasin.columns
#subbasin = gpd.read_file("C:/Users/langzx/OneDrive/AAMOSM2018/NewSubstoShare_Les3/New_subs1_LeS3model.shp")
subbasin.crs
subbasin.plot(color='white', edgecolor='grey')
stream = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_5_reach.shp")
f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_title('')
subbasin.plot(ax=ax, color='white', edgecolor='grey')
#stream.plot(ax=ax, color='#B3B3B3', edgecolor='grey')

p_names_sub = sub_result.columns.tolist()[:-1]
sns.color_palette("BrBG", 6)
# Create a dataset (fake)
#df = pd.DataFrame(np.random.random((10,10)), columns=["a","b","c","d","e","f","g","h","i","j"])
#sns.heatmap(df, cmap="BrBG")
pal = sns.color_palette("Greens",6)
color_ISOP = sns.color_palette("Oranges_r",3).as_hex()
color_ISOM = sns.color_palette("Greens_r",3).as_hex()
color_InCP =  sns.color_palette("Blues",6).as_hex()
color_InCM =  sns.color_palette("Purples",6).as_hex()
color_pal = color_ISOP + color_ISOM + color_InCP + color_InCM
color_pal
edgecolor='#B3B3B3'

legend_patches = []
for i in range(len(p_names_sub)):
    patch = mpatches.Patch(color=color_pal[i], label= p_names_sub[i])
    legend_patches.append(patch)
    



sub_merge_shape = pd.merge(subbasin, sub_result, how = 'left', left_on = 'Subbasin', right_on = 'subbasin') 
sub_merge_shape[sub_merge_shape[p_names_sub[15]] == 1].plot(color=color_pal[15], edgecolor='#B3B3B3', linewidth = 1)

p_names_sub[3]

f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_title('')
subbasin.plot(ax=ax, color='white', edgecolor= edgecolor)
for i in range(len(p_names_sub)):
    sub_merge_shape[sub_merge_shape[p_names_sub[i]] == 1].plot(ax = ax, 
                                                               color=color_pal[i], 
                                                              
                                                               edgecolor='#B3B3B3', 
                                                               legend = True,
                                                               linewidth = 0.4)



plt.legend(handles=legend_patches, loc='upper right)
plt.show()

MergeFile = []
for i in range(len(col_ld)):
    sub_new = sub_seedsmerge.groupby("subbasin")[col_ld[i]].apply(list).reset_index(name='new')
    sub_result = pd.DataFrame(sub_new['new'].to_list(), columns=practice)
    sub_result['subbasin'] = sub_new['subbasin'].str.extract('(\d+)').astype(int)
    sub_merge_shape = pd.merge(subbasin, sub_result, how = 'left', left_on = 'Subbasin', right_on = 'subbasin') 
    MergeFile.append(sub_merge_shape)
    

greek_letterz=[chr(code) for code in range(945,970)]
print(greek_letterz)

chr(955)
ld = [0.0, 0.5, 1.0]
ld_name = [chr(955) + '=' + str(i) for i in ld]
top_pct = [', Top10%', ', Top30%', ', Top60%']
plotname = [x + y for x in ld_name for y in top_pct]

plotname
def mapplot (merge_file, ax_num):
    subbasin.plot(ax=ax_num, color='white', edgecolor= edgecolor)
    for i in range(len(p_names_sub)):
        merge_file[merge_file[p_names_sub[i]] == 1].plot(ax = ax_num, 
                                                               color=color_pal[i], 
                                                              
                                                               edgecolor='#B3B3B3', 
                                                               legend = True,
                                                               linewidth = 0.4)
    ax_num.grid(False)
    ax_num.axis('off')
    



f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(72, 72))
ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
ax_list
mapplot (MergeFile[0], ax_list[0])
mapplot (MergeFile[1], ax_list[1])
mapplot (MergeFile[2], ax_list[2])
mapplot (MergeFile[3], ax_list[3])
mapplot (MergeFile[4], ax_list[4])
mapplot (MergeFile[5], ax_list[5])
mapplot (MergeFile[6], ax_list[6])
mapplot (MergeFile[7], ax_list[7])
mapplot (MergeFile[8], ax_list[8])
for i in range(9):
    ax_list[i].title.set_text(plotname[i])
plt.legend(handles=legend_patches, loc='upper right')
plt.show()        

    



plt.legend(handles=legend_patches)
plt.show()