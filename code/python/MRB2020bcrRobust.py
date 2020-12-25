import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from pathlib import Path
import itertools
import random
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data')
#data_folder = Path('/Users/ellelang/Documents/github/EAmosm/data')
dat = pd.read_csv (data_folder/'MRB2020/MRB1AAT1016.csv')
dat = dat.iloc[1:]
dat.shape

#dat = dat.replace(0, np.NaN)
random.seed(40)

dat['NitRed'] = np.where(dat['NitRed']==0, 
                        np.random.uniform(1, 100, size=len(dat)), 
                        dat['NitRed'])

dat['SedRed'] = np.where(dat['SedRed']==0, 
                        np.random.uniform(1, 100, size=len(dat)), 
                        dat['SedRed'])
dat['SED_BCR'] = dat['SedRed']/dat['Cost']
dat['NIT_BCR'] = dat['NitRed']/dat['Cost']
dat['DUO_BCR'] = 0.5*dat['SED_BCR'] + 0.5*dat['NIT_BCR']

dat['SED_BCR'] = dat['SED_BCR'] * 1000
dat['NIT_BCR']  = dat['NIT_BCR'] * 1000
dat['DUO_BCR'] = dat['DUO_BCR']* 1000
#####
sub_hru_id = pd.read_csv(data_folder/"MRB2020/sub_hru_all.csv")
#sub_hru_id = pd.read_csv(data_folder/"MRB2020/subbasin_ids.csv")
sub_hru_id.head(3)
#sub_hru_id['Practice'].unique()[12]
#sub_hru_id = sub_hru_id[sub_hru_id['Practice'] == ' InC-M25']
sub_datsmerge = pd.merge(sub_hru_id, dat, how = 'left', left_on = 'Label', right_on = 'Label') 
sub_datsmerge
sub_datsmerge = sub_datsmerge[(sub_datsmerge[['SED_BCR', 'NIT_BCR', 'DUO_BCR']] > 0).all(1)]
sub_datsmerge.shape
bcr_dis = sub_datsmerge.groupby(['subbasin'])['SED_BCR', 'NIT_BCR', 'DUO_BCR'].mean()
bcr_dis = bcr_dis.add_suffix(' ').reset_index()
subbasin = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_subbasins.shp")
subbasin.columns
sub_merge_bcr = pd.merge(subbasin, bcr_dis, how = 'left', left_on = 'Subbasin', right_on = 'subbasin') 

sub_merge_bcr['SED_BCR ']

streams = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/RiversMN.shp")
river_names = ['Minnesota River','Le Sueur River','Blue Earth River',\
              'Chippewa River' , 'Cottonwood River', 'Lac qui Parle River'
              ,'Redwood River','Pomme de Terre River' ,
             'Yellow Medicine River' ]

streams = streams.loc[streams['NAME'].isin (river_names)]
# "NAME" = 'Minnesota River' OR 

mbcrcomplete = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/ProjectedMinnesotaRiverBasinComplete.shp")
mbcrcomplete.name
mbcrcomplete = mbcrcomplete.to_crs("EPSG:4326")
#Blue Earth, Watonwan, Le Sueur
mbcrcomplete = mbcrcomplete[mbcrcomplete['name'].isin(['Blue Earth', 'Watonwan', 'Le Sueur'])]
mbcrcomplete = mbcrcomplete.to_crs("EPSG:4326")
mbcrcomplete.crs

plt.rcParams.update({'font.size': 13})

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(45, 15))
streams.plot(ax = ax[0], color = 'blue', legend = False, linewidth = 3.2)

sub_merge_bcr.plot(ax = ax[0], column = 'SED_BCR ', scheme = 'jenkscaspall', k = 12, cmap = "Greens", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      
#sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
#mbcrcomplete.plot(ax = ax[0], color = 'red',  alpha = 0.4)
ax[0].set_axis_off() 
ax[0].title.set_text(r'$\lambda = 1$')  
ax[0].title.set_fontsize(25)

streams.plot(ax = ax[1], color = 'blue', legend = False, linewidth = 3.2)
sub_merge_bcr.plot(ax = ax[1], column = 'DUO_BCR ', scheme = 'jenkscaspall', k = 12, cmap = "Greens", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      
#sub_30.plot(ax = ax[1], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
#
ax[1].set_axis_off() 
ax[1].title.set_text(r'$\lambda = 0.5$')  
ax[1].title.set_fontsize(25)

streams.plot(ax = ax[2], color = 'blue', legend = False, linewidth = 3.2)
sub_merge_bcr.plot(ax = ax[2], column = 'NIT_BCR ', scheme = 'jenkscaspall', k = 12, cmap = "Greens", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      
#sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
#
ax[2].set_axis_off() 
ax[2].title.set_text(r'$\lambda = 0$')  
ax[2].title.set_fontsize(25)
#fig.suptitle('Spatial pattern of wBCR ranking', size=28)
plt.savefig(data_folder/"MRB2020/wbcrspatial.png", dpi = 300)

#######



newdf = dat.sort_values(by=['SED_BCR'],ascending=False).reset_index()
newdf.shape
newdf['sedbcr_dist'] = abs(newdf['SED_BCR'].diff( periods = 1))
newdf['sedbcr_dist'][0] = 0
newdf['nitbcr_dist'] = abs(newdf['NIT_BCR'].diff( periods = 1))
newdf['nitbcr_dist'][0] = 0
newdf.columns


##################3
nsize = len (dat.index)
nsize
bcr_sc = dat['SED_BCR']
bcr_nc = dat['NIT_BCR']
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
ldname   
ld_df.head(3)
ld_df.shape
#ld_df.to_csv(data_folder/"MRB2020/df_lambda.csv",index = False)
df_c = pd.concat([dat.reset_index(drop=True), ld_df], axis=1)
df_c.shape
#df_c.to_csv(data_folder/"MRB2020/df_lambda.csv",index = False)
df_c.columns
colnamelist = ['ld0.0', 'ld0.5','ld1.0']
new_col_name = ['disld0', 'disld0.5', 'disld1']

# for i in range (3):
#     newdf_c = df_c.sort_values(by=colnamelist[i], ascending=False).reset_index()
#     df_c = newdf_c
#     df_c[new_col_name[i]] = abs(newdf_c[colnamelist[i]].diff( periods = 1)).fillna(0)

newdf_ld0 = df_c.sort_values(by='ld0.0', ascending=False).reset_index()
newdf_ld0['disld0'] = abs(newdf_ld0['ld0.0'].diff( periods = 1)).fillna(0)
newdf_ld05= df_c.sort_values(by='ld0.5', ascending=False).reset_index()
newdf_ld05['disld0.5'] = abs(newdf_ld05['ld0.5'].diff( periods = 1)).fillna(0)
newdf_ld1 = df_c.sort_values(by='ld1.0', ascending=False).reset_index()
newdf_ld1['disld1'] = abs(newdf_ld1['ld1.0'].diff( periods = 1)).fillna(0)

dw1d0 = newdf_ld0[['Label', 'disld0']]
dw1d05 = newdf_ld05[['Label', 'disld0.5']]
dw1d1 = newdf_ld1[['Label', 'disld1']]

dm0 = dw1d0.merge(dw1d05,on='Label').merge(dw1d1,on='Label')

dm0.columns
dm0.shape

new_df_c = df_c.merge(dm0, on = 'Label')

new_df_c.columns
new_df_c.to_csv(data_folder/"MRB2020/df_lambda.csv",index = False)

sub_hru_id = pd.read_csv(data_folder/"MRB2020/sub_hru_all.csv")
#sub_hru_id = pd.read_csv(data_folder/"MRB2020/subbasin_ids.csv")
#sub_hru_id.head(3)
sub_smerge = pd.merge(sub_hru_id, new_df_c, how = 'left', left_on = 'Label', right_on = 'Label') 
sub_smerge

#sub_smerge['subbasin'] = sub_smerge['subbasin'].str.extract('(\d+)').astype(int)
#sub_smerge.to_csv(data_folder/"MRB2020/subbasin_id_i.csv", index = False)
ave_dis = sub_smerge.groupby(['subbasin'])['disld0', 'disld0.5', 'disld1'].mean()
#ave_dis_nit =  sub_smerge.groupby(['subbasin'])[].mean()
ave_dis_df = ave_dis.add_suffix(' ').reset_index()
ave_dis_df['subbasin'] = ave_dis_df['subbasin'].astype(np.int64)
ave_dis_df.columns
ave_dis_df['disld0 '] = ave_dis_df['disld0 ']*1000
ave_dis_df['disld0.5 '] = ave_dis_df['disld0.5 ']*1000
ave_dis_df['disld1 '] = ave_dis_df['disld1 ']*1000
# #######################
subbasin = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_subbasins.shp")
subbasin.columns
subbasin['Subbasin']
sub_merge_robust = pd.merge(subbasin, ave_dis_df, how = 'left', left_on = 'Subbasin', right_on = 'subbasin') 
sub_merge_robust.columns
streams = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/RiversMN.shp")
#streams = streams.loc[streams.STRM_LEVEL == 3]
river_names = ['Minnesota River','Le Sueur River','Blue Earth River',\
              'Chippewa River' , 'Cottonwood River', 'Lac qui Parle River'
              ,'Redwood River','Pomme de Terre River' ,
             'Yellow Medicine River' ]

streams = streams.loc[streams['NAME'].isin (river_names)]
# "NAME" = 'Minnesota River' OR 
# "NAME" = 'Le Sueur River' OR  
# "NAME" = 'Blue Earth River' OR 
# "NAME" = 'Chippewa River' OR 
# "NAME" = 'Cottonwood River' OR 
# "NAME" = 'Lac qui Parle River' OR 
# "NAME" = 'Redwood River' OR 
# "NAME" = 'Pomme de Terre River' OR 
# "NAME" = 'Yellow Medicine River'

#['boxplot', 'equalinterval', 'fisherjenks', 'fisherjenkssampled', 'headtailbreaks', 'jenkscaspall', 'jenkscaspallforced', 'jenkscaspallsampled', 'maxp', 'maximumbreaks', 'naturalbreaks', 'quantiles', 'percentiles', 'stdmean', 'userdefined']

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(45, 15))
#sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[0], color = 'blue', legend = False, linewidth = 3.2)
sub_merge_robust.plot( ax = ax[0], column = 'disld1 ', scheme = 'jenkscaspall', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      

ax[0].set_axis_off() 
ax[0].title.set_text(r'$\lambda = 1$')  
ax[0].title.set_fontsize(25)


#sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[1], color = 'blue', legend = False, linewidth = 3.2)
sub_merge_robust.plot(ax = ax[1], column = 'disld0.5 ', scheme = 'jenkscaspall', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      

ax[1].set_axis_off() 
ax[1].title.set_text(r'$\lambda = 0.5$')  
ax[1].title.set_fontsize(25)

#streams.loc[streams.STRM_LEVEL == 2].head(2)
#sub_30.plot(ax = ax[0], linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
streams.plot(ax = ax[2], color = 'blue', legend = False, linewidth = 3.2)
sub_merge_robust.plot(ax = ax[2], column = 'disld0 ', scheme = 'jenkscaspall', k = 12, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True, linewidth = 0.2)      

ax[2].set_axis_off() 
ax[2].title.set_text(r'$\lambda = 0$')  
ax[2].title.set_fontsize(25)
#fig.suptitle('Spatial pattern of wBCR distance', size=28)
plt.savefig(data_folder/"MRB2020/robustspatial.png", dpi = 300, bbox_inches='tight')

#######################
selectedld = pd.read_csv(data_folder/"MRB2020/selectld.csv")
sub_hru_id = pd.read_csv(data_folder/"MRB2020/subbasin_ids.csv")
sub_hru_id.head(3)
sub_smerge = pd.merge(sub_hru_id, new_df_c, how = 'left', left_on = 'Label', right_on = 'Label') 
sub_smerge.columns
sub_smerge['ld0.5'].mean()
max_cost = sub_smerge.groupby(['subbasin'])['Cost'].mean()
#ave_dis_nit =  sub_smerge.groupby(['subbasin'])[].mean()
max_cost = max_cost.add_suffix(' ').reset_index()
max_cost['subbasin'] = ave_cost['subbasin'].astype(np.int64)

subbasin = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_5_subbasins.shp")
sub_cost = pd.merge(subbasin, max_cost, how = 'left', left_on = 'Subbasin', right_on = 'subbasin') 
sub_cost['Area_acre'] = (sub_cost['Area']/100)* 247.105
area_mean = np.mean([25, 50, 250, 500, 1000, 2000]) * 2.47105
area_mean
sub_cost['cost/acre'] = sub_cost['Cost']/area_mean
sub_cost['cost/acre'].mean()
plt.rcParams["legend.fontsize"] = 12

fig, ax = plt.subplots(1, figsize=(15, 15))
sub_cost.plot(ax = ax, column ='cost/acre', scheme = 'quantiles', k = 18, cmap = "YlOrBr", edgecolor = "#B3B3B3", legend= True)   

weighted_cost = pd.read_csv(data_folder/"MRB2020/weightedcost0923.csv")
weighted_cost.columns
weighted_cost['WLD_COST'] = weighted_cost['WLD_w'] + weighted_cost['ASC_w']
sub_cost = pd.merge(sub_cost,weighted_cost, how= 'left')
sub_cost.columns
sub_cost['surplus'] = sub_cost['cost/acre'] - weighted_cost['WLD_COST']
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='Payment > WTA')
grey_patch = mpatches.Patch(color='grey', label='Payment < WTA')

fig, ax = plt.subplots(1, figsize=(15, 15))
sub_cost.plot(ax = ax, linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
sub_cost[sub_cost['cost/acre'] < sub_cost['WLD_COST'] ].plot(ax = ax, color = 'grey', label = 'Cost < WTA')
sub_cost[sub_cost['cost/acre'] > sub_cost['WLD_COST'] ].plot(ax = ax, color = 'red', label = 'Cost > WTA')
plt.legend(plt.legend(handles=[red_patch, grey_patch]))

plt.rcParams["legend.fontsize"] = 6
plt.rcParams['savefig.dpi'] = 300

fig, ax = plt.subplots(1, figsize=(35, 35))
#sub_cost.plot(ax = ax, linewidth= 1.2,facecolor= "none", edgecolor='black', legend = False)
sub_cost.plot(column = 'surplus', scheme = 'jenkscaspall', k = 8, cmap = 'Reds',linewidth= 0.2,  edgecolor = "#B3B3B3", legend= True)
plt.savefig(data_folder/"MRB2020/surplus.png")

##############cost
subbasin = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/MRB_subbasins.shp")
subbasin.crs
#subbasin.crs = {'init': 'epsg:4326'}

plt.rcParams["legend.fontsize"] = 12
plt.rcParams['savefig.dpi'] = 300

subbasin.crs


counties_3states = gpd.read_file(data_folder/"MRB2020/shapefilesMRB/3statesMRBclipped.shp")
counties_3states.crs = {'init': 'epsg:4326'}
counties_3states['coords'] = counties_3states['geometry'].apply(lambda x: x.representative_point().coords[:])
counties_3states['coords'] = [coords[0] for coords in counties_3states['coords']]

counties_3states.NAME


weighted_cost = pd.read_csv(data_folder/"MRB2020/weightedcost0923.csv")
weighted_cost.columns
subweightedcost = pd.merge(subbasin,weighted_cost, how= 'left')

subweightedcost.crs
fig, ax = plt.subplots(1, figsize=(15, 15))

for idx, row in counties_3states.iterrows():
    ax.annotate(s=row['NAME'], xy=row['coords'],
                 verticalalignment='center',fontsize=15)
# Ot
fig, ax = plt.subplots(1, figsize=(15, 15))

subweightedcost.plot(ax = ax, linewidth= 1.2,edgecolor='black', legend = False)
#plt.show()


fig, ax = plt.subplots(1, figsize=(15, 15))
for idx, row in counties_3states.iterrows():
    ax.annotate(s=row['NAME'], xy=row['coords'],
                 verticalalignment='center',fontsize=15)

ax.legend(title='Weighted subbasin land opportunity cost')
subweightedcost.plot(ax = ax, column = 'WLD_w' ,scheme = 'jenkscaspall', k = 8, cmap = 'OrRd', linewidth= 1.2,edgecolor = "#B3B3B3", legend = True)
counties_3states.plot(ax = ax, edgecolor = "black", color = 'none', linewidth= 1.8)
#plt.legend(title="Weighted subbasin land opportunity cost")
plt.savefig(data_folder/"MRB2020/county_costs.png")
#plt.show()



fig, ax = plt.subplots(1, figsize=(15, 15))
counties_3states.plot(ax = ax, edgecolor = "#B3B3B3", color = 'lightgrey', linewidth= 0.8)


fig, ax = plt.subplots(1, figsize=(15, 15))
#subweightedcost.plot(ax = ax)
counties_3states.plot(ax = ax, edgecolor = "#B3B3B3", color = 'lightgrey', linewidth= 0.8)

