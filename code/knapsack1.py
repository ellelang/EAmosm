
import array
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
NGEN = 50

B = 2
N_SIZE = 10 

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
toolbox = base.Toolbox()


def randomgen(high, n):
    listrand = list(np.random.randint(high, size = n))
    return listrand
randomgen(2, 10)


creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox.register("attr_int", randomgen, B, N_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n = 20)
pop
ind1 = toolbox.individual()
ind1
ind1.fitness.valid # False



def evalKnapsack(individual):
    weight = np.sum(individual *ks["weight"])
    value1 = np.sum(individual *ks["value1"])
    value2 = np.sum(individual *ks["value2"])
    return weight, value1, value2


ind1.fitness.values = evalKnapsack(ind1)
ind1.fitness.values
np.sum(ks["weight"]*ind1)

for i in ind2:
    print(i)



#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxUniform,indpb = 0.5)    
#toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("mutate", tools.mutUniformInt, low = 0, up=1,indpb=0.2)
#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2, nd = "standard")

toolbox.register("evaluate", evalKnapsack)





##Mutation
mutant = toolbox.clone(ind1)
#ind2 = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
ind2, = tools.mutUniformInt(mutant, low = 0, up=1,indpb=0.5)
del mutant.fitness.values
ind2
ind1
ind2.fitness.values =  evalKnapsack(ind2)

ind2.fitness.values
ind1.fitness.values

##CROSSOVER
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxUniform(child1, child2, 0.5)
#tools.cxSimulatedBinary(child1, child2,0.5)

#tools.cxPartialyMatched(child1, child2)
del child1.fitness.values
del child2.fitness.values

selected = tools.selNSGA2([child1, child2], nd = "standard")
print (child1 in selected)	# True


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

pop

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_size = tools.Statistics(key=len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)
record = mstats.compile(pop)
print(record)



pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, verbose=True)
logbook
pop
logbook
#logbook = tools.Logbook()
#logbook.record(gen=0, evals=30, **record)
#logbook.header = "gen", "evals", "fitness", "size"
#logbook.chapters["fitness"].header = "min", "avg", "max"
#logbook.chapters["size"].header = "min", "avg", "max"
#print(logbook)

#logbook = tools.Logbook()
#logbook.record(gen=0, evals=30, **record)
logbook.header = "gen", "evals", "fitness", "size"
logbook.chapters["fitness"].header = "min", "avg", "max"
logbook.chapters["size"].header = "min", "avg", "max"


dflogbook = pd.DataFrame(logbook)
dflogbook.columns.values

gen = dflogbook["gen"]
fit_mins = dflogbook["min"]
#fit_mins = logbook.chapters["fitness"].select("min")
size_avgs =  dflogbook["avg"]
gen
fit_mins



import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()












print (ind2 is mutant)   # True
print (mutant is ind1)    # False


child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxBlend(child1, child2, 0.5)
del child1.fitness.values
del child2.fitness.values

selected = tools.selBest([child1, child2], 2)
print (child1 in selected)	# True

def evaluate(individual):
    # Do some hard computing on the individual
    a = sum(individual)
    b = len(individual)
    return a, 1. / b

ind1.fitness.values = evaluate(ind1)
print (ind1.fitness.valid)    # True
print (ind1.fitness)          # (2.73, 0.2)

mutant = toolbox.clone(ind1)
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
del mutant.fitness.values

print (ind2 is mutant)   # True
print (mutant is ind1)    # False


child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxBlend(child1, child2, 0.5)
del child1.fitness.values
del child2.fitness.values

selected = tools.selBest([child1, child2], 2)
print (child1 in selected)	# True

selected = toolbox.select(pop, LAMBDA)
offspring = [toolbox.clone(ind) for ind in selected]
offspring

pop = toolbox.population(n=20)
pop





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

pop

#
#stats = tools.Statistics(lambda ind: ind.fitness.values)
#stats.register("avg", numpy.mean, axis=0)
#stats.register("std", numpy.std, axis=0)
#stats.register("min", numpy.min, axis=0)
#stats.register("max", numpy.max, axis=0)
#
#record = stats.compile(pop)
#record


stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_size = tools.Statistics(key=len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)
record = mstats.compile(pop)
print(record)

logbook = tools.Logbook()
logbook.record(gen=0, evals=30, **record)
logbook.header = "gen", "evals", "fitness", "size"
logbook.chapters["fitness"].header = "min", "avg", "max"
logbook.chapters["size"].header = "min", "avg", "max"
print(logbook)


pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, verbose=True)
logbook.header = "gen", "avg", "spam"
logbook

gen = logbook.select("gen")
fit_mins = logbook.chapters["fitness"].select("min")
size_avgs = logbook.chapters["size"].select("avg")

fit_mins
size_avgs

gen = logbook.select("gen")
fit_mins = logbook.chapters["fitness"].select("min")
size_avgs = logbook.chapters["size"].select("avg")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()



#################
##################
###############################
#################
##################
###############################
#################
##################
###############################
# knapsack with 10 objects A, B, ..., J with weight w_i and value v_i
Ob_name = [chr(x) for x in range(65, 75)]
Ob_w = random.sample(range(1, 20), 10)
Ob_value1 = random.sample(range(1, 50), 10)
Ob_value2 = random.sample(range(1, 50), 10)

a = np.random.randint(2, size=10)


dict_new = {
    'object': Ob_name,
    'weight': Ob_w,
    'value1':Ob_value1,
    'value2':Ob_value2
}

ks = pd.DataFrame(dict_new)
ks


items = {}
# Create random items and store them in the items' dictionary.
for i in range(10):
    items[i] = ( random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100))
items
MAX_ITEM = 10000
MAX_WEIGHT = 100000
def Fitness(x): 
    val1 = np.sum(x*ks["value1"])
    val2 = np.sum(x*ks["value2"])
    weight = -1 * np.sum(x * ks['weight'])
    if weight > ks_limit:
        return 0
    
    elif x[1] * x[3] * x[9] == 1:
        
        return [-10,-10, weight]
    else:
        return [val1,val2,weight]

def evalKnapsack(individual):
    weight = 0.0
    value1 = 0.0
    value2 = 0.0
    for item in individual:
        weight += items[item][0]
        value1 += items[item][1]
        value2 += items[item][2]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value1, value2


offspring = toolbox.select(pop, len(pop))
offspring

    # Clone the selected individuals
offspring = map(toolbox.clone, offspring)
offspring 
# Apply crossover on the offspring
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values




def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2


def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(10))
    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    random.seed(64)
    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    main()