from __future__ import (absolute_import, division, print_function)
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('bmh')
from pathlib import Path

data_folder = Path('D:/UWonedrive/OneDrive - UW/AAMOSM2018/0828mapdata')
from shapely.geometry import Point, Polygon
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame

#############

from matplotlib.colors import LinearSegmentedColormap

def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])




cmapgrey = grayscale_cmap('viridis')




##################



subbasin = gpd.read_file(data_folder/"subbasins.shp")	
subbasin['geometry'].head()
#subbasin['geometry'] = subbasin['geometry'].to_crs(epsg=4326)
# destination coordinate syste
subbasin.crs = {'init': 'epsg:4326'}
stream =  gpd.read_file(data_folder/ "LeSueur_Streams.shp")
stream.crs= {'init' :'epsg:4326'}
gage = gpd.read_file(data_folder/"gage2.shp")
gs = GeoSeries([Point(-120, 45), Point(-121.2, 46), Point(-122.9, 47.5),Point(-122.9, 47.5),Point(-122.9, 47.5)])
gage.crs = {'init' :'epsg:4326'}
wcmo = gpd.read_file(data_folder/"WCMO_project.shp")	

subbasin['coords'] = subbasin['geometry'].apply(lambda x: x.representative_point().coords[:])
subbasin['coords'] = [coords[0] for coords in subbasin['coords']]
#####################
f, ax = plt.subplots(1, figsize=(20, 20))
ax.set_title('')
for idx, row in subbasin.iterrows():
    ax.annotate(s=row['Subbasin'], xy=row['coords'],
                 verticalalignment='center',fontsize=15)
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3', cmap='summer_r'
subbasin.plot(ax = ax, column = 'watershed', linewidth=0.8, cmap='gray',edgecolor='#B3B3B3', legend = True)
stream.plot(ax = ax, edgecolor='black')
gage.plot(ax = ax, marker='*', color='black', markersize=400)

#wcmo.plot(ax = ax, edgecolor = 'darkorchid')
ax.grid(False)
ax.axis('off')

#######################
cmap1=cmapgrey
cmap1

#cmap1 = plt.cm.summer_r
cmap2 = plt.cm.Paired
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(42, 42))
fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
# cmap = summer_r
subbasin.plot(ax = ax1, column = 'watershed', linewidth=0.8, cmap=cmap1,edgecolor='#B3B3B3', legend = True)
stream.plot(ax = ax1, edgecolor='black')
gage.plot(ax = ax1, marker='*', color='darkgrey', markersize=800)
for idx, row in subbasin.iterrows():
    ax1.annotate(s=row['Zone'], xy=row['coords'],
                 verticalalignment='center',fontsize=20,arrowprops=dict(facecolor='black', shrink=0.05))
#ax1.set_title('LeSueur River Watershed and hydrologic subbasins')
custom_lines1 = [Line2D([0],[0], color='w', lw=20),
                Line2D([0], [0], color=cmap1(0.), lw=20),
                Line2D([0], [0], color=cmap1(.5), lw=20),
                Line2D([0], [0], color=cmap1(1.), lw=20),
                Line2D([0], [0], marker='*', color='w',markersize=20, markerfacecolor='darkgrey'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w'),
                Line2D([0], [0],  color='w',markersize=20, markerfacecolor='w')]
legend1 = ax1.legend(custom_lines1, ['Subwatersheds:','Cobb River', 'LeSueur River', 'Maple River','Gages','Zone 1 = Upland','Zone 2 = Transitional','Zone 3 = Incised'], fontsize=20)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_linewidth(0.0)
#ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, 
#            size=60, weight='bold')
#ax1.grid(False)
#ax1.axis('off')
plt.savefig('intromapLE.png', bbox_inches='tight', dpi=500)

subbasin.plot(ax=ax2, column = 'Zone', linewidth=0.8,  cmap='Paired', edgecolor='dimgray')
stream.plot(ax = ax2, edgecolor='blue')
wcmo.plot(ax = ax2, color = 'darkorchid', edgecolor = 'darkorchid')
for idx, row in subbasin.iterrows():
    ax2.annotate(s=row['Subbasin'], xy=row['coords'],
                 verticalalignment='center',fontsize=20)
    
custom_lines2 = [Line2D([0],[0], color='w', lw=20),
                Line2D([0], [0], color=cmap2(0.), lw=20),
                Line2D([0], [0], color=cmap2(.5), lw=20),
                Line2D([0], [0], color=cmap2(1.), lw=20),
                Line2D([0], [0], marker='s', color='w',markersize=20, markerfacecolor='darkorchid')]
legend2 = ax2.legend(custom_lines2, ['Geomorphic Zones:', 'Zone 1 = Upland','Zone 2 = Transitional','Zone 3 = Incised', 'Potential sites for WCMO'], fontsize=15)   
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_linewidth(0.0)

ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, 
            size=60, weight='bold')

ax2.grid(False)
ax2.axis('off')
#ax2.set_title('Geomorphic Zones and water storage sites')
plt.savefig('intromap.png', bbox_inches='tight', dpi=200)