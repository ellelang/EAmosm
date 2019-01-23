
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

demodataseeds = pd.read_csv(data_folder/"demodata_seeds.csv")
nsize = 50
nsize
Groundset = range(nsize)
Groundset
Subsets   = range(3)
Ob_w = demodataseeds['SRed']
Ob_value1 = demodataseeds['Duck']
Ob_value2 = demodataseeds['Cost']

Set = [Ob_w, Ob_value1, Ob_value2]
Set
SetObjPriority = [1, 1, 1]
SetObjWeight   = [0.0, 0.9, -1.0]

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
