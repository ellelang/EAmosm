
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
from operator import attrgetter
ldfront = pd.read_csv(data_folder/"ldfront_seeds.csv")
sed_sum = ldfront['SRed']
duck_sum = ldfront['Duck']
cost_sum = ldfront['Cost']



demodataseeds = pd.read_csv(data_folder/"demodata_seeds.csv")

seeds_ld = [[0] * nsize for j in range(len(ldtopname))]
for j in range(len(ldtopname)):
    seeds_ld[j] = demodataseeds[ldtopname[j]]
seeds_ld

seeds_array = np.array(seeds_ld)
seeds_array
seeds_array_to_list = seeds_array.tolist()
json_file = "ldseeds.json" 
json.dump(seeds_array_to_list, open(data_folder/json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)


nsize = 50
nsize
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.2
NGEN = 20
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

def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)



def initIndividual(icls, content):
    return icls(content)



toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, data_folder/"ldseeds.json")

pop = toolbox.population_guess()
pop
hof = tools.ParetoFront()
creator.create("FitMulti", base.Fitness, weights=(0.5, 0.5, -1.0))
creator.create("Individual", array.array, typecode='d', fitness= creator.FitMulti)
toolbox.register("attr_int", randomgen, 2, nsize)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#pop = toolbox.population(n = N_POP)

toolbox.register("mate", tools.cxUniform,indpb = 0.5)    
toolbox.register("mutate", tools.mutUniformInt, low = 0, up=1,indpb=0.2)
toolbox.register("select", tools.selNSGA2, nd = "standard")
toolbox.register("evaluate", evamultiobj)
ind1 = toolbox.individual()
ind1
ind1.fitness.valid # False

ind1.fitness.values = evamultiobj(ind1)
ind1.fitness.values




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

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10)
algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
                              cxpb=0.5, mutpb=0.2, ngen=150, 
                              stats=stats, halloffame=hof)
pop, hof = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
                              cxpb=0.5, mutpb=0.2, ngen=50, 
                              halloffame=hof)
best = max(pop, key=attrgetter("fitness"))
print (best,best.fitness)
list(best)

pop

for i in range(nsize):
    if (sum(pop[i]) <= 40):
        pop
#hof = tools.ParetoFront()
#pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
hof[0]
sum(pop[49])
pop.sort(key=lambda x: x.fitness.values)
sum(pop[3])
pop[0] #the worst solution
sum(pop[-1]) # the best solution   
front = np.array([ind.fitness.values for ind in pop])
front
front[-1,1]
front[-1][1]
front = np.array([ind.fitness.values for ind in pop if sum(ind)< topval])
front
front.shape
front[-1,0]
front[-1,1]
front[-1,2]
front[0,0]
front[0,1]
front[0,2]

logbook

weight_f = front[:,0]
value1_f = front[:,1]
value2_f = front[:,2]
ks
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(weight_f, value1_f, value2_f)
ax.set_xlabel('sediment')
ax.set_ylabel('duck')
ax.set_zlabel('cost')


##################loop through ld
ngen=10
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
def evamultiobj(individual):
    sred = np.sum(individual *ks["SRed"])
    duck = np.sum(individual *ks["Duck"])
    cost = np.sum(individual *ks["Cost"])
    return sred, duck, cost

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, data_folder/"ldseeds.json")
pop = toolbox.population_guess()

def ealdseed (ldval):
    creator.create("FitMulti", base.Fitness, weights=(ldval, 1-ldval, -1.0))
    creator.create("Individual", array.array, typecode='d', fitness= creator.FitMulti)
    toolbox.register("attr_int", randomgen, 2, nsize)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
    toolbox.register("mate", tools.cxUniform,indpb = 0.5)    
    toolbox.register("mutate", tools.mutUniformInt, low = 0, up=1,indpb=0.2)
    toolbox.register("select", tools.selNSGA2, nd = "standard")
    toolbox.register("evaluate", evamultiobj)
    pop = toolbox.population_guess() 
    #hof = tools.ParetoFront()
    for g in range(ngen):
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
    
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10)
#    pop, hof = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
#                              cxpb=0.5, mutpb=0.2, ngen=20, 
#                              halloffame=hof)
    pop.sort(key=lambda x: x.fitness.values)
    front = np.array([ind.fitness.values for ind in pop])
    sed,duck,cost = front[-1,0], front[-1,1], front[-1,2]
    
    return sed, duck, cost

#for ind in pop:
#    if (sum (ind) < 30):
#        print (sum(ind))

#sed_dsum = [0] * len(ldtopname)
#duck_dsum = [0] * len(ldtopname)
#cost_dsum = [0] * len(ldtopname)
sed_dsum = [0] * len(ld)
duck_dsum = [0] * len(ld)
cost_dsum = [0] * len(ld)

import time
start = time.time()
"the code you want to test stays here"
for i in range (len (ld)):
    ld_val = ld[i]
    result = ealdseed (ld_val)
    sed_dsum [i] = result [0]
    duck_dsum [i] = result [1]
    cost_dsum [i] = result [2]
end = time.time()
print(end - start)

#import time
#start = time.time()
#"the code you want to test stays here"
#for i in range (len (ld)):
#    ld_val = ld[i]
#    for t in range (len (top)):
#        top_val = int(round(top[t] * 100 ))
#        result = ealdseed (ld_val, top_val)
#        sed_dsum [nameindex[i]+t] = result [0]
#        duck_dsum [nameindex[i]+t] = result [1]
#        cost_dsum [nameindex[i]+t] = result [2]
#end = time.time()
#print(end - start)

dict_dfront = {
    'SRed': sed_dsum,
    'Duck': duck_dsum ,
    'Cost': cost_dsum 
}
sed_dsum
frontd_df = pd.DataFrame(dict_dfront)
frontd_df
frontd_df.to_csv(data_folder/"deapfront.csv",index = False, sep=',', encoding='utf-8')


ldfront = pd.read_csv(data_folder/"ldfront_seeds.csv")
sed_sum = ldfront['SRed']
duck_sum = ldfront['Duck']
cost_sum = ldfront['Cost']

gurobifront = pd.read_csv(data_folder/"gurobifront.csv")
sed_gsum = gurobifront['SRed']
duck_gsum = gurobifront['Duck']
cost_gsum = gurobifront['Cost']




from mpl_toolkits import mplot3d


fig = plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')
ax.scatter3D(sed_sum, duck_sum, cost_sum, c= cost_sum,  cmap = 'autumn', label = "BCR front")
ax.scatter3D(sed_gsum, duck_gsum, cost_gsum, color = 'darkgreen', label = "gurobi front")
ax.scatter3D(sed_dsum, duck_dsum, cost_dsum, cmap = 'viridis', label = "EA front")
ax.set_xlabel('Sediment')
ax.set_ylabel('Duck')
ax.set_zlabel('Cost')
ax.legend()
fig.savefig(data_folder/'Front_dgld.pdf') 
