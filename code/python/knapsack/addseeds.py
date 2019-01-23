
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
import random
import json

from math import sqrt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import itertools
demodata = pd.read_csv(data_folder/"demodata.csv")
rankbyld0 = demodata["rankbyld0.0"]
rankbyld0


ld = np.round(np.arange(0,1.1,0.1),1)
ld
ldname = ["ld" + str(i) for i in ld]
ldname

ldrankname = ["rankbyld" + str(i) for i in ld]
ldrankname


top = np.round(np.arange(0.1,1,0.1),1)
top_pct = top*100
topname = ["top" + str(int(i)) for i in top_pct]
topname

ldtopname = [x + y for x in ldname for y in topname]
ldtopname
len(ldtopname)
nameindex = list (range(0,99,9))
nameindex
ldtopname[98]


nsize = len(demodata.index)
nsize


for i in range (len (ld)):
    rankby = demodata[ldrankname[i]]
    for t in range (len (top)):
        topby = int(round(top[t] * nsize ))
        seed_ld = [0]*50 
        for j in range (nsize) :
            if rankby [j] <= topby:
                seed_ld [j] = 1
            else:
                seed_ld [j] = 0
        demodata[ldtopname[nameindex[i]+t]] = seed_ld
          
demodata.to_csv(data_folder/"demodata_seeds.csv",index = False, sep=',', encoding='utf-8')



sed_sum = [0] * len(ldtopname)
duck_sum = [0] * len(ldtopname)
cost_sum = [0] * len(ldtopname)




for i in range (len(ldtopname)):
    sed_sum [i] = np.sum(demodata['SRed'] * demodata[ldtopname[i]])
    duck_sum [i] = np.sum(demodata['Duck'] * demodata[ldtopname[i]])
    cost_sum [i] = np.sum(demodata['Cost'] * demodata[ldtopname[i]])
    
sed_sum

dict_front = {
    'SRed': sed_sum,
    'Duck': duck_sum ,
    'Cost': cost_sum 
}

front_df = pd.DataFrame(dict_front)
front_df
front_df.to_csv(data_folder/"ldfront_seeds.csv",index = False, sep=',', encoding='utf-8')


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(sed_sum, duck_sum, cost_sum, cmap='viridis')
ax.set_xlabel('Sediment')
ax.set_ylabel('Duck')
ax.set_zlabel('Cost')

import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Scatter3d(
    x= sed_sum, y=duck_sum, z=cost_sum,
    marker=dict(
        size=4,
        color=cost_sum,
        colorscale='Viridis',
    ),
    line=dict(
        color='#1f77b4',
        width=1
    )
)

data = [trace]

layout = dict(
    width=800,
    height=700,
    autosize=False,
    title='ld bcr',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=-1.7428,
                y=1.0707,
                z=0.7100,
            )
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='ld', height=700)

top[1] * nsize 
int(round(top[1] * nsize ))
seed_ld = [0]*50 



for i in range (len(ld)):
    for t in range (len(top)):
        print (ldtopname[nameindex[i]+t])

for i in range (nsize):
    if (rankbyld0[i] <= int(round(top[1] * nsize ))):
        seed_ld[i] = 1
    else:
        seed_ld[i] = 0

seed_ld     
