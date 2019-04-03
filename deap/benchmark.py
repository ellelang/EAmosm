import random
import array
import numpy as np
import json
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import benchmarks  
from deap.benchmarks.tools import diversity, convergence, hypervolume
from scipy.special import comb
import numpy.matlib
import itertools
# M = 3 number of objectives M = 2, 3 OR M = 6, 10
# N_S = 100 * M: THE NUMBER OF DECISION VARIABLES N_S = 200, 300 OR 600, 1000
# N_K =  THE number of subcomponents in each variable group = 5 
#% H1, H2 contains the parameter for the two-layer simplex lattice design up
#% to 15 objectives, you may also add new settings by yourself
#% The number of reference vectors (i.e., population size N) to be generated is determined by h1, h2 and M
#H1 = [99 13  7  5  4  3  3  3  3  2  2  2  2  2]; h1 = H1(M-1);
#H2 = [ 0  0  0  0  1  2  2  2  2  2  2  2  2  2]; h2 = H2(M-1);
#N = nchoosek(h1+M-1,M-1) + nchoosek(h2+M-1,M-1); % the number of reference vectors
#k = find(~isstrprop(Problem,'digit'),1,'last');
#D = D = 10; %number of decision variables
M = 3
popsize = 100
range (M)
n_k = 5
n_s = M * 100
a, c0 = 3.8, 0.1

chaos = lambda c: a * c* (1 - c)
c = chaos (c0)
C = [c]
for i in range (M-1):
    c = chaos (c)
    C.insert(0, c)   
    
C[0:M]
La = []
Lb = []
NNg = np.array(0)
NNg = np.ceil(np.round(C/np.sum(C)*n_s))
NNg

# number of decision variables
N_ns = np.sum (NNg) * n_k
D = int((M - 1) + N_ns)
D
half_b = M - 1
half_b
half_r = int(D - M + 1)
half_r
lu_row1 = np.concatenate((np.zeros(half_b), 0 * np.ones(half_r, dtype = int)))   
lu_row1
lu_row2 = np.concatenate((np.zeros(half_b), 10 * np.ones(half_r, dtype = int)))   
lu_row2
## boundary of decision variables
lu = np.array([lu_row1, lu_row2])
lu
np.shape(lu)
Bounday = lu[[1,0]]
Bounday
Bounday[1,:]
La = 1 + np.arange(M,D+1)/D
np.shape(La)
La
Lb = 1 + np.cos(0.4 * np.pi * np.arange(M,D+1)/D)
Lb


#Correlation matrix
Aa = np.identity(M)
Ab = np.identity(M)
Aa
for i in range (M - 1):
    Aa[i,i+1] = 1
Aa
Ac = np.ones((M,M))

Population = np.random.rand(popsize, int(D))

p1 = Population * np.matlib.repmat(Bounday[1,:], popsize, 1)
np.shape(p1)
p2 = (1- Population) * np.matlib.repmat(Bounday[0,:], popsize, 1)

Output = p1 + p2
np.shape(p1+p2)

np.shape(La)
end = int(D+1)
end
begin = M - 1
np.shape(Output)
popeval = Output
popevalnew = popeval
popeval
popevalnew[:, begin:end] = popeval[:,begin:end] * np.matlib.repmat(La, popsize, 1)\
 -   10 * np.tile(np.matrix(Population[:,1]).transpose(), (1, half_r))
 

##############s

Xf = popeval[:,0:M-1]
Xf
Xs= np.empty(np.shape(popevalnew))
temparray = np.ndarray(shape = (M,), dtype = "object")

for i in range (M):
    if i > 0:
        idx_xs1 = int( M + n_k * np.sum (NNg[:i]))
    else:
        idx_xs1 = int(M)
    idx_xs2 = int(idx_xs1 + n_k * NNg[i] - 1)
    Xs[:,idx_xs1:idx_xs2] = popevalnew[:, idx_xs1: idx_xs2]
    temparray[i] = popevalnew[:, idx_xs1: idx_xs2]

np.shape(temparray)
np.shape(temparray[0])

Xs   
np.shape(Xs)
Xs
arr = np.concatenate(temparray[0]).astype('int')
arr
arr = np.vstack([np.hstack(c) for c in temparray[0]])
np.shape(arr)


############Sampling true PF AND PS
### W = T_uniform(k,m)
a = np.arange(1, M)
a
H = np.floor((n_k*a.prod())**(1/(M-1)))
while comb(H + M-1, M - 1, exact=True) >= n_k and H > 0:
    H = H - 1
if comb(H + M-1, M - 1) <= 2 or H == 0:
    H = H + 1
n_k= int(comb(H + M-1, M - 1))
n_k
a1 = np.arange(M-1)
a1
H
#np.array(list(itertools.combinations(np.arange(1, H + M), M -1)))
temp = np.array(list(itertools.combinations(np.arange(1, H + M), M -1))) - np.matlib.repmat(a1, int(comb(H + M-1, M - 1)), 1) - 1
temp[:,0]
W = np.zeros((n_k,M))
W[:,2]
W[:,1] = temp[:,1] - 0
#W[:,2]= temp[:,2] - temp[:,1]
for i in range (1,M):
    print(i) 
    W[:,i]= temp[:,i-1]
W[:,-1] = H - temp[:,-1]
W = W/H
W

###W = T_repeat (k,M)

###################basic function values
# Evaluate the individuals with an invalid fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.sphere)
segXss
invalid_ind = [ind for ind in segXss]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
fitnesses
for ind, fit in zip(invalid_ind, fitnesses):
     ind.fitness.values = fit
G = np.zeros((popsize,M))
G
NNg[0]
for i in range (M):
    g1 = 0
    g2 = 0
    nss = NNg [i]
    segXss = temparray[i]
    for j in range (n_k):
        Xss = segXss[:, int((j - 1) * nss + 1) : int(j * nss)]
        g1 =  g1 + sphere_func(Xss)/nss
        g2 =  g2 + sphere_func(Xss)/nss
    if i % 2 == 1:
        G[:, i] = g1
    else:
        G[:, i] = g2
G = G/n_k


##########objective values
F = np.zeros((popsize,M))
F
for i in range (M):
    Gi = G * Aa[i, :]
    F[:,i] = ((1 + Gi ).T * np.prod (Xf[:, :M-1-i], axis = 1)).T[:,i]
    if i > 0:
        F [:, i] = F [:, i] * (1 - Xf[:, (M-i)])
Output2 = F





Aa[1,:]
Gi =  1 + G * Aa[1, :]
np.shape(Gi)
((1 + Gi ).T * np.prod (Xf[:, :M-2+1], axis = 1)).T
F [:, 1]* Xf[:, (M-2 + 1)] 
    
np.shape(np.prod (Xf[:, :M-2+1], axis = 1))


sphere_func(Xss)


idx_xs1
idx_xs2
np.shape(popevalnew)
np.shape(popevalnew[:,idx_xs1:idx_xs2])
Population [:,1]
M
half_r
np.shape(Population[:,1].transpose())
aaa= 
np.shape(aaa)
aaa

popeval[:, 1]

popsize
b = np.array([[1,4],[3,1]])


b
b.sort(axis = 0)
b
comb(10, 3, exact=True)
np.floor(-23.11)
np.zeros((3,5))
def sphere_func (x):
    a = x ** 2
    fit = a.sum(axis=1)
    return fit


def schwefel_func (x):
    
  a = benchmarks.schwefel(x)
  fit = a.max (axis = 1)
  return fit
