from pathlib import Path
#data_folder = Path("C:/Users/langzx/OneDrive/AAMOSM2018/maps/0425bcrSurvive")
#D:\UWonedrive\OneDrive - UW\AAMOSM2018
data_folder = Path("D:/UWonedrive/OneDrive - UW/AAMOSM2018/maps/0425bcrSurvive")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import SeabornFig2Grid as sfg

datadistr = pd.read_csv(data_folder/'MOSM distributionlist.csv')

datadistr.columns.values

datadistr.head()

datald_1 = datadistr.loc[datadistr['ld'] == 1]
datald_05 = datadistr.loc[datadistr['ld'] == 0.5]
datald_0 = datadistr.loc[datadistr['ld'] == 0]

sns.set(style="dark")
sns.palplot(sns.color_palette("Greys"))
#sns.palplot(sns.color_palette("Blues"))
plt.legend(loc='upper left')
sns.set_context("poster")   

g0 = sns.catplot(x="Type", y="Distribution", hue="CostLevel", data=datald_1,
               height=5, kind="bar", palette = "Greys")
g0.set(ylim=(0, 1))
g0.set_ylabels("Distribution of Selected MOs")
g0.set_xlabels("")
g0.set_titles('$\lambda = 1$')
g0.fig.text(0.35, 0.8,'$\lambda = 1$', fontsize=25) 

g1 = sns.catplot(x="Type", y="Distribution", hue="CostLevel", data=datald_05,
               height=5, kind="bar", palette = "Greys")

g1.set_titles('')
g1.set(ylim=(0, 1))
g1.set_ylabels(' ')  
g1.set_xlabels("")
g1.fig.text(0.35, 0.8,'$\lambda = 0.5$', fontsize=25) #add text



g2 = sns.catplot(x="Type", y="Distribution", hue="CostLevel", data=datald_0,
                height=5, kind="bar", palette = "Greys",legend_out=False)
g2.set_titles('$\lambda = 0$')
g2.set(ylim=(0, 1))
g2.set_ylabels(' ')  
g2.set_xlabels("")
g2.fig.text(0.35, 0.8,'$\lambda = 0$', fontsize=25) 

fig = plt.figure(figsize=(30,15))
gs = gridspec.GridSpec(1,3)
mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
mg0.fig.text(0.15, 0.8,'$\lambda = 1$', fontsize = 25) 
mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
mg1.fig.text(0.45, 0.8,'$\lambda = 0.5$', fontsize = 25) 

mg2 = sfg.SeabornFig2Grid(g2, fig, gs[2])
mg2.fig.text(0.75, 0.8,'$\lambda = 0$', fontsize = 25) 

gs.tight_layout(fig)

#gs.update(top=0.7)

plt.show()
fig.savefig('distributionplot_greys.pdf',dpi = 300)   

##############EXAMPLES

