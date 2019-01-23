
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json
import array
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

ldfront = pd.read_csv(data_folder/"ldfront_seeds.csv")
sed_sum = ldfront['SRed']
duck_sum = ldfront['Duck']
cost_sum = ldfront['Cost']


demodataseeds = pd.read_csv(data_folder/"demodata_seeds.csv")
nsize = 50
nsize
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.2
NGEN = 250
N_SIZE = 50
N_POP = 500
MU = 50

ID = demodataseeds['ID']
Ob_s = demodataseeds['SRed']
Ob_d = demodataseeds['Duck']
Ob_c = demodataseeds['Cost']
dict_new = {
    'ID': ID,
    'SRed': Ob_s,
    'Duck':Ob_d,
    'Cost':Ob_c
}

ks = pd.DataFrame(dict_new)
ks


ld = np.round(np.arange(0,1.1,0.1),1)
ld
top = np.round(np.arange(0.1,1,0.1),1)
topby = int(round(top[2] * nsize ))
topby

ldname = ["ld" + str(i) for i in ld]
top_pct = top*100
topname = ["top" + str(int(i)) for i in top_pct]
topname
ldtopname = [x + y for x in ldname for y in topname]
ldtopname

nameindex = list (range(0,99,9))
nameindex

def randomgen(high, n):
    listrand = list(np.random.randint(high, size = n))
    return listrand
a = randomgen(2, 50)
a
def evamultiobj(individual):
    sred = np.sum(individual *ks["SRed"])
    duck = np.sum(individual *ks["Duck"])
    cost = np.sum(individual *ks["Cost"])
    return sred, duck, cost

toolbox = base.Toolbox()
creator.create("FitnessMulti", base.Fitness, weights=(0.5, 0.5, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
toolbox.register("attr_int", randomgen, 2, nsize)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n = N_POP)




toolbox.register("mate", tools.cxUniform,indpb = 0.5)    
toolbox.register("mutate", tools.mutUniformInt, low = 0, up=1,indpb=0.2)
toolbox.register("select", tools.selNSGA2, nd = "standard")
toolbox.register("evaluate", evamultiobj)
#pop
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
record = stats.compile(pop)
record

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"



for g in range(NGEN):
    # Select and clone the next generation individuals
    offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))

    # Apply crossover and mutation on the offspring
    
    offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats)
#hof = tools.ParetoFront()
#pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

pop.sort(key=lambda x: x.fitness.values)
pop
pop[0] #the worst solution
pop[-1] # the best solution   
front = np.array([ind.fitness.values for ind in pop])
front
front.shape
front[-1,0]
front[-1,1]
front[-1,2]
front[0,0]
front[0,1]
front[0,2]

logbook

weight_f = front[-1,0]
value1_f = front[-1,1]
value2_f = front[-1,2]
ks
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(weight_f, value1_f, value2_f)
ax.set_xlabel('sediment')
ax.set_ylabel('duck')
ax.set_zlabel('cost')