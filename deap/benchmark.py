import random
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
D = (M - 1) + N_ns

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

popsize
b = np.array([[1,4],[3,1]])


b
b.sort(axis = 0)
b
comb(10, 3, exact=True)
np.floor(-23.11)
np.zeros((3,5))
def sphere_func (x):
    a = benchmarks.sphere(x)
    fit = a.sum(axis=1)
    return fit


def schwefel_func (x):
    
  a = benchmarks.schwefel(x)
  fit = a.max (axis = 1)
  return fit
