from __future__ import (absolute_import, division, print_function)
import itertools
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('bmh')
from pathlib import Path
data_folder = Path('D:/OneDrive/AAMOSM2018/0828mapdata')
from shapely.geometry import Point, Polygon
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame

subbasin = gpd.read_file(data_folder/"subbasins.shp")	
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
# Other nice categorical color maps (cmap) include 'Set2' and 'Set3'
subbasin.plot(ax = ax, column = 'watershed', linewidth=0.8, cmap='summer_r',edgecolor='#B3B3B3', legend = True)
stream.plot(ax = ax, edgecolor='blue')
gage.plot(ax = ax, marker='*', color='red', markersize=400)

#wcmo.plot(ax = ax, edgecolor = 'darkorchid')
ax.grid(False)
ax.axis('off')

#######################
cmap1 = plt.cm.summer_r
cmap2 = plt.cm.Paired
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(42, 42))

subbasin.plot(ax = ax1, column = 'watershed', linewidth=0.8, cmap='summer_r',edgecolor='#B3B3B3', legend = True)
stream.plot(ax = ax1, edgecolor='blue')
gage.plot(ax = ax1, marker='*', color='red', markersize=400)
for idx, row in subbasin.iterrows():
    ax1.annotate(s=row['Subbasin'], xy=row['coords'],
                 verticalalignment='center',fontsize=20)
#ax1.set_title('LeSueur River Watershed and hydrologic subbasins')
custom_lines1 = [Line2D([0],[0], color='w', lw=20),
                Line2D([0], [0], color=cmap1(0.), lw=20),
                Line2D([0], [0], color=cmap1(.5), lw=20),
                Line2D([0], [0], color=cmap1(1.), lw=20),
                Line2D([0], [0], marker='*', color='w',markersize=20, markerfacecolor='r')]
legend1 = ax1.legend(custom_lines1, ['Subwatersheds:','Cobb River', 'LeSueur River', 'Mapple River','Gages'], fontsize=15)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_linewidth(0.0)
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, 
            size=60, weight='bold')
ax1.grid(False)
ax1.axis('off')


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