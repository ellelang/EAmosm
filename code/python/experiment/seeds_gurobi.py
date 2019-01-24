
import random
from gurobipy import *
from __future__ import print_function
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
import random


ldfront = pd.read_csv(data_folder/"ldfront_seeds.csv")
sed_sum = ldfront['SRed']
duck_sum = ldfront['Duck']
cost_sum = ldfront['Cost']


demodataseeds = pd.read_csv(data_folder/"demodata_seeds.csv")
nsize = 50
nsize
Groundset = range(nsize)
Groundset
Subsets   = range(3)
ID = demodataseeds['ID']
Ob_w = demodataseeds['SRed']
Ob_value1 = demodataseeds['Duck']
Ob_value2 = demodataseeds['Cost']

Set = [Ob_w, Ob_value1, Ob_value2]
Set
SetObjPriority = [1, 1, 1]
SetObjWeight   = [0.0, 0.9, -1.0]
top = np.round(np.arange(0.1,1,0.1),1)
topby = int(round(top[2] * nsize ))
topby
ld = np.round(np.arange(0,1.1,0.1),1)
ld

ldname = ["ld" + str(i) for i in ld]
top_pct = top*100
topname = ["top" + str(int(i)) for i in top_pct]
topname
ldtopname = [x + y for x in ldname for y in topname]
ldtopname

nameindex = list (range(0,99,9))
nameindex

def opt_exct (ldval, topval,costcons) :
    SetObjPriority = [0, 0, 0]
    SetObjWeight   = [-ldval, -(1-ldval), -1.0]
    model = Model('multiobj')
    Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')
    cost_cons = quicksum(Elem[k]*Set[2][k] for k in range(len(Elem)))
    model.addConstr(Elem.sum() <= topval)
    model.addConstr(cost_cons <= costcons)
    model.ModelSense = GRB.MINIMIZE
    model.setParam(GRB.Param.PoolSolutions, 100)
    for i in Subsets:
        objn = sum(Elem[k]*Set[i][k] for k in range(len(Elem)))
        model.setObjectiveN(objn, i, SetObjPriority[i], SetObjWeight[i],
                            1.0 + i, 0.01, 'Set' + str(i))

    model.write('bcrmultiobj.lp')
    model.optimize()
    obj1 = model.getObjective(0)
    obj2 = model.getObjective(1)
    obj3 = model.getObjective(2)
    sedsum = obj1.getValue()
    ducksum = obj2.getValue()
    costsum = obj3.getValue()
    return sedsum, ducksum,costsum

sed_gsum = [0] * len(ldtopname)
duck_gsum = [0] * len(ldtopname)
cost_gsum = [0] * len(ldtopname)

for i in range (len (ld)):
    ld_val = ld[i] 
    for t in range (len (top)):
        top_val = int(round(top[t] * nsize ))
        cost_val = cost_sum[nameindex[i]+t]
        result = opt_exct (ld_val, top_val, cost_val)
        sed_gsum[nameindex[i]+t] = result [0]
        duck_gsum[nameindex[i]+t] = result [1]
        cost_gsum[nameindex[i]+t] = result [2]
        
dict_gfront = {
    'SRed': sed_gsum,
    'Duck': duck_gsum ,
    'Cost': cost_gsum 
}

frontg_df = pd.DataFrame(dict_gfront)
frontg_df
frontg_df.to_csv(data_folder/"gurobifront.csv",index = False, sep=',', encoding='utf-8')

from mpl_toolkits import mplot3d


fig = plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')
ax.scatter3D(sed_sum, duck_sum, cost_sum, c= cost_sum,  cmap = 'autumn', label = "BCR front")
ax.scatter3D(sed_gsum, duck_gsum, cost_gsum, color = 'darkgreen', label = "gurobi front")
ax.set_xlabel('Sediment')
ax.set_ylabel('Duck')
ax.set_zlabel('Cost')
ax.legend()
fig.savefig(data_folder/'Front_gurobi_ldseed.pdf') 


model = Model('multiobj')
Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')

model.addConstr(Elem.sum() <= 5, name='top10')

model.ModelSense = GRB.MAXIMIZE

    # Limit how many solutions to collect
model.setParam(GRB.Param.PoolSolutions, 100)

    # Set and configure i-th objective
for i in Subsets:
    objn = sum(Elem[k]*Set[i][k] for k in range(len(Elem)))
    model.setObjectiveN(objn, i, SetObjPriority[i], SetObjWeight[i],
                       1.0 + i, 0.01, 'Set' + str(i))

model.write('bcrmultiobj.lp')
model.optimize()
model.X
var = model.getVars()
var[0].x 

for i in range(len(var)):
    if var[i].x == 1:
        print (var[i])

demodataseeds['ID'].loc[demodataseeds['ld0.0top10'] == 1]

subdata = demodataseeds.loc[demodataseeds['ld0.5top10'] == 1]
sed_bcr= np.sum(subdata['SRed']*subdata['ld0.5top10'])
duck_bcr = np.sum(subdata['Duck']*subdata['ld0.5top10'])
cost_bcr = np.sum(subdata['Cost']*subdata['ld0.5top10'])

obj1 = model.getObjective(0)
obj2 = model.getObjective(1)
obj3 = model.getObjective(2)

sed_bcr
print(obj1.getValue())
duck_bcr
print(obj2.getValue())
cost_bcr
print(obj3.getValue())
