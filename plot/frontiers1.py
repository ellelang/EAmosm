from __future__ import (absolute_import, division, print_function)
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import xlrd
data_folder = Path('D:/UWonedrive/OneDrive - UW/AAMOSM2018/ECR/Rplotresults')

###bcr_uniform#
#rm (list = ls())
#bcr_uniform099 <-  read.xlsx("0429results.xlsx", sheet = "0420 1610 bcr uni 099")
#
##uniformseed099#
#uniformseed099 <-  read.xlsx("0429results.xlsx", sheet = "4-24 2110 vanilla same pop bcr")
#
##Evaluated_wBCR#
#Evaluated_wBCR <- read.xlsx("0429results.xlsx", "bcr seeds")
#
##noseeds#
#noseeds <-  read.xlsx("0429results.xlsx", sheet = "lastgen4301248unseeded_099cr")

 
dfbcr_uniform = pd.read_excel(open(data_folder/'0429results.xlsx','rb'), sheetname='0420 1610 bcr uni 099')
dfbcr_uniform.columns
dfbcr_uniform.shape[0]
ft_bcr_uniform = dfbcr_uniform[["Outlet Sediment", "Duck Increase", "Cost"]]
name_bcr_uniform = ['wBCR_Uniform'] * dfbcr_uniform.shape[0]
name_bcr_uniform 
ft_bcr_uniform['Frontiers'] = name_bcr_uniform 
ft_bcr_uniform['Frontiers']

df_uniform = pd.read_excel(open(data_folder/'0429results.xlsx','rb'), sheetname='4-24 2110 vanilla same pop bcr')
df_uniform.columns
ft_uniform = df_uniform[["Outlet Sediment", "Duck Increase", "Cost"]]
name_uniform = ['Uniform'] * df_uniform.shape[0]
ft_uniform['Frontiers'] = name_uniform
ft_uniform['Frontiers']


df_bcreval = pd.read_excel(open(data_folder/'0429results.xlsx','rb'), sheetname='bcr seeds')
df_bcreval.columns
ft_bcreval = df_bcreval[["Outlet Sediment", "Duck Increase", "Cost"]]
name_bcreval = ['wBCR_eval'] * df_bcreval.shape[0]
ft_bcreval['Frontiers'] = name_bcreval
ft_bcreval['Frontiers']



df_noseed = pd.read_excel(open(data_folder/'0429results.xlsx','rb'), sheetname='lastgen4301248unseeded_099cr')
df_noseed.columns
ft_noseed = df_noseed[["Outlet Sediment", "Duck Increase", "Cost"]]
name_noseed = ['Noseed'] * df_noseed.shape[0]
ft_noseed['Frontiers']  = name_noseed

pd1 = pd.concat([ft_bcr_uniform,ft_uniform,ft_bcreval,ft_noseed], sort = True)
pd1['Cost($Million/Year)'] = pd1['Cost']/1000000
pd1.columns
pd1['Frontiers']



#plt.style.use('grayscale')
sns.set(font_scale=1.5)
sns.set_palette("gray")

g1 = sns.lmplot(x="Outlet Sediment", y="Cost($Million/Year)", hue="Frontiers",\
                truncate=True,  data=pd1,fit_reg=False,legend_out=False, \
                height=10, markers=['o', '<','s','d'],aspect=1.2)

#g1 = sns.lmplot(x="Cost($Million/Year)", y="Sediment.Reduction", hue="EA frontier",  data=pd1)
g1.ax.legend(loc=1,title = "Pareto-frontiers Comparison",frameon=True)
g1.ax.grid(False)



g2 = sns.lmplot(x="Duck Increase", y="Cost($Million/Year)", hue="Compare",\
                truncate=True,  data=pd1,fit_reg=False,legend_out=False, legend = False, \
                height=10, markers=['o', '<','s','d'],aspect=1.2)

g2.ax.grid(False)
g2.set(xlabel="Duck hatchlings")
#g2.ax.xlabel("Duck hatchlings")

fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(1,2)
mg0 = sfg.SeabornFig2Grid(g1, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g2, fig, gs[1])

gs.tight_layout(fig)

plt.show()
fig.savefig('Pareto_frontiers.pdf',dpi = 300)   

#################for 3d
#from matplotlib import interactive
#interactive(True)

df_3d = pd.read_csv(data_folder/"for3d.csv")
df_3d.columns
df_3d.head
fig3d = plt.figure(figsize=(15,15))
df_3d_wbcr = df_3d[df_3d['origin'] == 'WBCR']
df_3d_wbcr_eval = df_3d[df_3d['origin'] == 'WBCR_evaluated']

ax = plt.axes(projection='3d')
ax.scatter3D(df_3d_wbcr.Sediment, df_3d_wbcr.Duck, df_3d_wbcr.Cost, \
             color = '#17202A', label = "wBCR", marker = 'o',
             linewidth=4)
ax.scatter3D(df_3d_wbcr_eval.Sediment, df_3d_wbcr_eval.Duck, df_3d_wbcr_eval.Cost,\
             color = '#909497', label = "wBCR_evaluated", marker = 's',
             linewidth=4)
ax.azim = 60
ax.elev = 30
#ax.dist = 100
ax.set_xlabel('Sediment Reduction',labelpad=20)
ax.set_ylabel('Duck hatchlings',labelpad=20)
ax.set_zlabel('Cost ($/Year)',labelpad=20)
ax.legend(loc=2, fontsize = 16)
plt.show()
ax.figure.savefig('Front_dgld.pdf') 