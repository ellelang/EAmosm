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
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.2
NGEN = 250
N_SIZE = 10
N_POP = 500
MU = 50
# knapsack with 10 objects A, B, ..., J with weight w_i and value v_i
Ob_name = [chr(x) for x in range(65, 75)]
Ob_w = random.sample(range(1, 20), 10)
Ob_value1 = random.sample(range(1, 50), 10)
Ob_value2 = random.sample(range(1, 50), 10)

dict_new = {
    'object': Ob_name,
    'weight': Ob_w,
    'value1':Ob_value1,
    'value2':Ob_value2
}

ks = pd.DataFrame(dict_new)
ks
def randomgen(high, n):
    listrand = list(np.random.randint(high, size = n))
    return listrand
a = randomgen(2, 10)
a

toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox.register("attr_int", randomgen, 2, N_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n = N_POP)
pop



def evalKnapsack(individual):
    weight = np.sum(individual *ks["weight"])
    value1 = np.sum(individual *ks["value1"])
    value2 = np.sum(individual *ks["value2"])
    if individual[1] * individual[3] * individual[9] == 1:
        return 2, 1, value2
    return weight, value1, value2

ind1 = toolbox.individual()
ind1
ind1.fitness.valid # False

ind1.fitness.values = evalKnapsack(ind1)
ind1.fitness.values

toolbox.register("mate", tools.cxUniform,indpb = 0.5)    
toolbox.register("mutate", tools.mutUniformInt, low = 0, up=1,indpb=0.2)
toolbox.register("select", tools.selNSGA2, nd = "standard")
toolbox.register("evaluate", evalKnapsack)


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
    
    
#pop
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

record = stats.compile(pop)
record
print(record)
logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"
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
front[:,0]
front[:,1]
front[:,2]
logbook

weight_f = front[:,0]
value1_f = front[:,1]
value2_f = front[:,2]
ks
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(weight_f, value1_f, value2_f)
ax.set_xlabel('Weight')
ax.set_ylabel('Value1')
ax.set_zlabel('Value2')
#ax.contour3D(weight_f, value1_f, value2_f, 50, cmap='binary')