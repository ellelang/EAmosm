from __future__ import (absolute_import, division, print_function)
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
# The two statemens below are used mainly to set up a plotting
# default style that's better than the default from Matplotlib 1.x
# Matplotlib 2.0 supposedly has better default styles.
import seaborn as sns
plt.style.use('bmh')
from pathlib import Path
#data_folder = Path('C:/Users/langzx/OneDrive/AAMOSM2018/maps/0425bcrSurvive')
data_folder = Path('D:/OneDrive/AAMOSM2018/maps/0425bcrSurvive')

from shapely.geometry import Point, Polygon
from matplotlib.lines import Line2D
import pandas as pd

import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
#import pyepsg
mpl.__version__, pd.__version__, gpd.__version__

from shapely.wkt import loads


#colormap = mpl.cm.Dark2.colors   # Qualitative colormap
subbasin = gpd.read_file("D:/OneDrive/AAMOSM2018/NewSubstoShare_Les3/New_subs1_LeS3model.shp")

#subbasin = gpd.read_file("C:/Users/langzx/OneDrive/AAMOSM2018/NewSubstoShare_Les3/New_subs1_LeS3model.shp")
subbasin.crs
subbasin.plot(color='white', edgecolor='grey')

stream =  gpd.read_file("D:/OneDrive/AAMOSM2018/0828mapdata/LeSueur_Streamsproject.shp")

MO = ['TLMO','WCMO','RAMO','ICMO','NCMO', 'AFMO','BFMO']
colormap = ['dodgerblue','darkorchid','maroon','olive','sienna','orange','green']
Level = [10, 30, 65]
LD = [1,0.5,0]
filename = ["shapefile_" + str(i) for i in MO]
filename
levelname = ["X" + str(i) + '_' for i in Level]
levelname
ldname = ["lambda = " + str(i) for i in LD]
ldname
ldvalue = ldname[0]

f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_title('')
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'
subbasin.plot(ax=ax, color='white', edgecolor='grey')
stream.plot(ax=ax, color='blue', edgecolor='grey')
for i in range(len(MO)):
    path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
    vars()[filename[i]] = gpd.read_file(data_folder/path)
    if i < 4:
        vars()[filename[i]][vars()[filename[i]]['X65_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)
    else:
        vars()[filename[i]][vars()[filename[i]]['X65_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 5)
    #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
    #vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)

ax.grid(False)
ax.axis('off')
plt.legend()
plt.savefig('submap.pdf', bbox_inches='tight', dpi = 500)    



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(42, 42))

subbasin.plot(ax=ax1, color='white', edgecolor='grey')
for i in range(len(MO)):
    path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
    vars()[filename[i]] = gpd.read_file(data_folder/path)
    #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
    vars()[filename[i]][vars()[filename[i]]['X10_'] == 4].plot(ax=ax1, color = colormap[i], label = MO[i], edgecolor = colormap[i], linewidth = 1 )
ax1.grid(False)
ax1.axis('off')

subbasin.plot(ax=ax2, color='white', edgecolor='grey')
for i in range(len(MO)):
    path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
    vars()[filename[i]] = gpd.read_file(data_folder/path)
    #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
    vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax2, color = colormap[i], label = MO[i],edgecolor = colormap[i], linewidth = 1)
ax2.grid(False)
ax2.axis('off')


subbasin.plot(ax=ax3, color='white', edgecolor='grey')
for i in range(len(MO)):
    path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
    vars()[filename[i]] = gpd.read_file(data_folder/path)
    #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
    vars()[filename[i]][vars()[filename[i]]['X65_'] == 4].plot(ax=ax3, color = colormap[i], label = MO[i], edgecolor = colormap[i], linewidth = 1)
ax3.grid(False)
ax3.axis('off')



def submaps (ldvalue, **args):
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(42, 42))
    
    subbasin.plot(ax=ax1, color='white', edgecolor='black')
    stream.plot(ax=ax1, color='blue', edgecolor='grey')

    for i in range(len(MO)):
        path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
        vars()[filename[i]] = gpd.read_file(data_folder/path)
        if i < 4:
            vars()[filename[i]][vars()[filename[i]]['X10_'] == 4].plot(ax=ax1, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 3)
        else:
            vars()[filename[i]][vars()[filename[i]]['X10_'] == 4].plot(ax=ax1, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 4)
        #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
        #vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)
    
    ax1.grid(False)
    ax1.axis('off') 
    
    subbasin.plot(ax=ax2, color='white', edgecolor='black')
    stream.plot(ax=ax2, color='blue', edgecolor='grey')

    for i in range(len(MO)):
        path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
        vars()[filename[i]] = gpd.read_file(data_folder/path)
        if i < 4:
            vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax2, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 2)
        else:
            vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax2, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 4)
        #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
        #vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)
    
    ax2.grid(False)
    ax2.axis('off')
    
    subbasin.plot(ax=ax3, color='white', edgecolor='black')
    stream.plot(ax=ax3, color='blue', edgecolor='grey')

    for i in range(len(MO)):
        path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
        vars()[filename[i]] = gpd.read_file(data_folder/path)
        if i < 4:
            vars()[filename[i]][vars()[filename[i]]['X65_'] == 4].plot(ax=ax3, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)
        else:
            vars()[filename[i]][vars()[filename[i]]['X65_'] == 4].plot(ax=ax3, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 4)
        #if len(vars()[filename[i]][vars()[filename[i]]['X10_'] == 4]['X10_']) > 0:
        #vars()[filename[i]][vars()[filename[i]]['X30_'] == 4].plot(ax=ax, color = colormap[i], label = MO[i], edgecolor = colormap[i],linewidth = 1)
    
    ax3.grid(False)
    ax3.axis('off')
    
    
################################

plt.figure(figsize=(50,50))
submaps(ldname[0])
plt.savefig('submap1.png', bbox_inches='tight', dpi=200)

submaps(ldname[1])
plt.savefig('submap0.5.png', bbox_inches='tight', dpi=200)

submaps(ldname[2])
plt.savefig('submap0.png', bbox_inches='tight', dpi=200)

import matplotlib.image as mpimg 
#read image

img1 = mpimg.imread('submap1.png')
img2 = mpimg.imread('submap0.5.png')
img3 = mpimg.imread('submap0.png')
#plot image (2 subplots)
text1 = "Cost level ~ 10% sediment reduction"
text2 = "Cost level ~ 30% sediment reduction"
text3 = "Cost level ~ 65% sediment reduction"



custom_lines = [Line2D([0], [0], color=colormap[0], lw=4),
                Line2D([0], [0], color=colormap[1], lw=4),
                Line2D([0], [0], color=colormap[2], lw=4),
                Line2D([0], [0], color=colormap[3], lw=4),
                Line2D([0], [0], color=colormap[4], lw=4),
                Line2D([0], [0], color=colormap[5], lw=4),
                Line2D([0], [0], color=colormap[6], lw=4)]


fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, sharex = True, sharey=True, figsize = (15,15))
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
plt.grid(False)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.set_title('$\lambda$ = 1')
ax2.set_title('$\lambda$ = 0.5')
ax3.set_title('$\lambda$ = 0')
legend = ax1.legend(custom_lines, MO, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=1,fontsize=14)
legend.get_frame().set_facecolor('white')
plt.figtext(0.12, 0.1, text1, fontsize=14)
plt.figtext(0.42, 0.1, text2, fontsize=14)
plt.figtext(0.70, 0.1, text3, fontsize=14)
plt.savefig('mapplot.png', bbox_inches='tight', dpi=500)





#######################################
submaps(ldname[2])



ldvalue = ldname[2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(42, 42))

subbasin.plot(ax=ax1, color='white', edgecolor='grey')

path = ldvalue + '/' + 'MOSed/' + MO[3] + '.shp'
vars()[filename[3]] = gpd.read_file(data_folder/path)
    #if len(vars()[filename[3]][vars()[filename[3]]['X10_'] == 4]['X10_']) > 0:
vars()[filename[3]][vars()[filename[3]]['X10_'] == 4].plot(ax=ax1, color=colormap[3], edgecolor = colormap[3], label = MO[3], linewidth = 3)
ax1.grid(False)
ax1.axis('off')













ldname[1]
plt.plot([0,1,2],[0,2*i,2*i], color=color,label=MO[i])


for i in range(len(MO)):
    path = ldvalue + '/' + 'MOSed/' + MO[i] + '.shp'
    vars()[filename[i]] = gpd.read_file(data_folder/path)
    
vars()[filename[]]
## Lambda = 1
f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_title('')
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'
subbasin.plot(ax=ax, color='white', edgecolor='grey')
vars()[filename[6]][vars()[filename[6]]['X10_'] == 4].plot(ax=ax, color = 'red',linewidth = 15)
ax.grid(False)
ax.axis('off')
plt.axis('equal');


















#shapefile_NCMO = shapefile_NCMO.to_crs(epsg=4269)
#outfp = data_folder / 'lambda = 1/MOSed/NCMO.shp'
# Save to disk
#shapefile_NCMO.to_file(outfp)
#shapefile_NCMO
#10% sediment 

shapefile_NCMO = gpd.read_file(data_folder/'lambda = 1/MOSed/NCMO.shp')
shapefile_AFMO = gpd.read_file(data_folder/'lambda = 1/MOSed/AFMO.shp')
shapefile_BFMO = gpd.read_file(data_folder/'lambda = 1/MOSed/BFMO.shp')
shapefile_WCMO = gpd.read_file(data_folder/'lambda = 1/MOSed/WCMO.shp')
shapefile_ICMO = gpd.read_file(data_folder/'lambda = 1/MOSed/ICMO.shp')
shapefile_RAMO = gpd.read_file(data_folder/'lambda = 1/MOSed/RAMO.shp')
shapefile_TLMO = gpd.read_file(data_folder/'lambda = 1/MOSed/TLMO.shp')

shapefile_NCMO.columns.values
shapefile_AFMO.columns.values
shapefile_BFMO.columns.values
shapefile_WCMO.columns.values
shapefile_ICMO.columns.values
shapefile_RAMO.columns.values 
shapefile_TLMO.columns.values

type(shapefile_AFMO)
shapefile_NCMO[shapefile_NCMO['X10_'] == 4].head()





shapefile_NCMO[shapefile_NCMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_AFMO[shapefile_AFMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_BFMO[shapefile_BFMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_WCMO[shapefile_WCMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_ICMO[shapefile_ICMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_RAMO[shapefile_RAMO['X10_'] == 4].plot(figsize=(8, 8))
shapefile_TLMO[shapefile_TLMO['X10_'] == 4].plot(figsize=(8, 8))






# 30% sediment cost

# 65% sediment cost



## Lambda = 0.5



# lambda = 01















GeoSeries([loads('POINT(1 2)'), loads('POINT(1.5 2.5)'), loads('POINT(2 3)')])

subbasin = gpd.read_file(data_folder/"subbasins.shp")	
subbasin_p4 =  subbasin.to_crs(epsg=4326)
subbasin.crs
subbasin.crs = from_epsg(4326)
subbasin_p4 =  subbasin.to_crs(epsg = 4326)
subbasin.crs = {'init' :'epsg:4326'}
subbasin.crs
subbasin['geometry']

#subbasin.plot( color='red', markersize=100, figsize=(4, 4))
gage = gpd.read_file(data_folder/"gage2.shp")
gs = GeoSeries([Point(-120, 45), Point(-121.2, 46), Point(-122.9, 47.5),Point(-122.9, 47.5),Point(-122.9, 47.5)])
#subbasin['Watershed_Zone'] = subbasin.watershed.map(str) + " " + subbasin.Zone

#subbasin = gpd.read_file("C:\\Users\langzx\\Onedrive\\AAMOSM2018\\newmapdata\\subs1.shp")
stream =  gpd.read_file(data_folder/ "LeSueur_Streams.shp")
## project
subbasin.crs = {'init' :'epsg:4326'}
subbasin.crs
#subbasin_p4 =  subbasin.to_crs(epsg=4326)
gage.crs = {'init' :'epsg:4326'}

#gage_p4.plot(marker='*', color='red', markersize=100, figsize=(4, 4))

#gage2_p4 =  gage2.to_crs
stream.crs= {'init' :'epsg:4326'}

len(subbasin.columns.values)
subbasin.columns.values
type(subbasin['geometry'])

subbasin['coords'] = subbasin['geometry'].apply(lambda x: x.representative_point().coords[:])
subbasin['coords'] = [coords[0] for coords in subbasin['coords']]


f, ax = plt.subplots(1, figsize=(20, 20))
ax.set_title('')


# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'
subbasin.plot(ax = ax, column = 'watershed', linewidth=0.8, cmap='summer_r',edgecolor='#B3B3B3', legend = True)
stream.plot(ax = ax, edgecolor='blue')
gage.plot(ax = ax, marker='*', color='red', markersize=400)
ax.set_ylim([43.6, 44.3])
plt.axis('equal');

##Annotation
for idx, row in subbasin.iterrows():
    ax.annotate(s=row['Zone'], xy=row['coords'],
                 verticalalignment='center')

#plt.axis('equal');
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=20),
                Line2D([0], [0], color=cmap(.5), lw=20),
                Line2D([0], [0], color=cmap(1.), lw=20),
                Line2D([0], [0], marker='*', color='w',markersize=20, markerfacecolor='r'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w')]
legend = ax.legend(custom_lines, ['COB', 'LES', 'MAP','Gages','Zone 1 = Upland','Zone 2 = Transitional','Zone 3 = Incised'], fontsize=20)
legend.get_frame().set_facecolor('white')


type(subbasin.ix[23, 'geometry'])
subbasin['coords'] = subbasin['geometry'].apply(lambda x: x.representative_point().coords[:])
subbasin['coords'] = [coords[0] for coords in subbasin['coords']]

subbasin.plot( cmap='summer_r', legend=True, k=3, figsize=(12, 15))
for idx, row in subbasin.iterrows():
    plt.annotate(s=row['Zone'], xy=row['coords'],
                 horizontalalignment='center')
gage.plot( marker='*', color='red', markersize=200) 
stream.plot(ax=ax, edgecolor='blue')



subbasin.plot(column='watershed', cmap='summer_r',  k=3, figsize=(8, 10));

subbasin.plot(color='white', edgecolor='black', figsize=(8, 8));


# Visualize
ax = subbasin.plot()


gdf.columns.values
gdf.plot(cmap='Set2', figsize=(10, 10));

ax = subbasin.plot(color='white', edgecolor='black')
ax.set_axis_off()
gdf.plot(ax = ax, edgecolor='white')
plt.show()

##################

plt.figure(figsize=(20,5))
cmap = plt.cm.summer_r
t = ("Zone 1 = Upland\n"
     "Zone 2 = Transitional\n"
     "Zone 3 = Incised")

f, ax = plt.subplots(1, figsize=(20,20))
ax.set_facecolor('white')
ax.set_ylim([43.6, 44.3])
ax.grid(False)
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'

type(subbasin.ix[23, 'geometry'])
subbasin['coords'] = subbasin['geometry'].apply(lambda x: x.representative_point().coords[:])
subbasin['coords'] = [coords[0] for coords in subbasin['coords']]


subbasin.plot(ax=ax,column='watershed',linewidth=0.8, cmap='summer_r',edgecolor='#B3B3B3', legend = True)
gage.plot(ax = ax, marker='*', color='red', markersize=200)
stream.plot(ax = ax, edgecolor='blue')

for idx, row in subbasin.iterrows():
    ax.annotate(s=row['Zone'], xy=row['coords'],
                 verticalalignment='center')

#plt.axis('equal');
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=20),
                Line2D([0], [0], color=cmap(.5), lw=20),
                Line2D([0], [0], color=cmap(1.), lw=20),
                Line2D([0], [0], marker='*', color='w',markersize=20, markerfacecolor='r'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w')]
legend = ax.legend(custom_lines, ['COB', 'LES', 'MAP','Gages','Zone 1 = Upland','Zone 2 = Transitional','Zone 3 = Incised'], fontsize=20)
legend.get_frame().set_facecolor('white')

#ax.text(0.97, 0.87, t,
#       verticalalignment='top', horizontalalignment='right',
#        transform=ax.transAxes,
#        fontsize=15,wrap=True)
#ax.text(0.5,0.5,u'\u25B2 \nN ', ha='center', fontsize=20, family='Arial')
plt.savefig('mosmmap.pdf', bbox_inches='tight')





subbasin.crs
# Visualize
ax = subbasin.plot()


gdf.columns.values
gdf.plot(cmap='Set2', figsize=(10, 10));

ax = subbasin.plot(color='white', edgecolor='black')
ax.set_axis_off()
gdf.plot(ax = ax, edgecolor='white')
plt.show()

