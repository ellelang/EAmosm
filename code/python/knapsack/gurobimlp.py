
import random
from gurobipy import *
from __future__ import print_function


Groundset = range(10)
Subsets   = range(3)
Ob_w = random.sample(range(1, 20), 10)
Ob_value1 = random.sample(range(1, 50), 10)
Ob_value2 = random.sample(range(1, 50), 10)
SetObjPriority = [0, 0, 0]
SetObjWeight   = [1.0, 0.25, 0.75]
model = Model('multiobj')

Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')
x1,x3,x9 = Elem[1],Elem[3],Elem[9]
model.addConstr(x1== or_(x3,x9), name='eps1')
