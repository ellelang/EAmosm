import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import scipy.special as sps
from deap import tools
from deap import creator
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import benchmarks
from deap import base
from deap import algorithms
from pathlib import Path
import seaborn as sns
data_folder = Path('C:/Users/langzx/Desktop/github/EAmosm/data')
maple29 = pd.read_csv (data_folder/"maple29.csv")
maple29.head()
cost_s = maple29['Cost'].sample(n=100, replace = False, random_state=1)

#############################################


ms = np.array([0.71, 0.731, 0.581, 0.684, 0.626, 0.611, 0.66, 0.56])
vs = np.array([0.11, 0.114, 0.16, 0.126, 0.068, 0.07, 0.07, 0.137])
shapes = ms/vs
scales = ms**2/vs
shapes[0]
scales


s = np.random.gamma(shapes[0], scales[0], 100)
dat =  pd.DataFrame({'Sratio': s,
                     'Cost' : cost_s,
                     'bcr' : s/ cost_s,
                     'ID': np.arange(1, 101, 1)}).reset_index(drop = True)

NBR_ITEMS = 100
top = np.arange(2, 102, 2)
topname = ["Top" + str(i) for i in top]
topname
for t in range(len(top)):
    dat[topname[t]] = [0] * NBR_ITEMS
    dat.loc[dat.nlargest(top[t], 'bcr').index, topname[t]] = 1

dat
dat.columns
# where is the top 5 largest bcr
dat.nlargest(25, ['bcr']).index
# where is the top 5 largest SedRed
dat.nlargest(5, ['Sratio']).index

sed = dat['Sratio']
#cost = [1] * NBR_ITEMS
cost = dat['Cost']
siteid = dat['ID']


sed_noepis = [sum(sed * dat[i]) for i in topname]
cost_noepis = [sum(cost * dat[i]) for i in topname]
sed_noepis
plt.scatter(sed_noepis, cost_noepis, c='m', marker='D', label='no_epistasis')
# save seeds
seedslist = []
for i in range(len(top)):
    indexvalues = dat.index[dat[topname[i]] == 1].tolist()
    seedslist.append(indexvalues)
np.array(seedslist)
json_file = "bcrseeds_gamma100.json"
json.dump(seedslist, open(data_folder/json_file, 'w',
                          encoding='utf-8'), sort_keys=True, indent=4)



######## add interactions
# top 25 --> id 74
# top 45 --> id 68
# top 65 --> id 65
# top 85 --> id 38
dat['rankbcr'] = dat['bcr'].rank(ascending=0)
dat


#dat.loc[dat.nlargest(20, 'bcr').index, 'sed_epi'] = 

s1 = np.random.gamma(shapes[1], scales[1], 100)
s1[20:25]
s2 = np.random.gamma(shapes[2], scales[2], 100)
s2[25:45] 
s3 = np.random.gamma(shapes[3], scales[3], 100)
s3[45:65]
s4 = np.random.gamma(shapes[4], scales[4], 100) 
s4[65:85]
s5 = np.random.gamma(shapes[5], scales[5], 100) 
s5[85:100]


set0 = set(dat.nlargest(20, ['bcr']).index)
set0
set1 = set(dat.nlargest(25, ['bcr']).index)
set2 = set(dat.nlargest(45, ['bcr']).index)
set3 = set(dat.nlargest(65, ['bcr']).index)
set4 = set(dat.nlargest(85, ['bcr']).index)
set5 = set(dat.nlargest(100, ['bcr']).index)

rank_first = list(set0)
rank_first
rank_a = list(set(set1) ^ set(set0))
rank_a
rank_b = list(set(set1) ^ set(set2))
rank_b
rank_c = list (set(set3) ^ set(set2))
rank_c 
rank_d = list (set(set4) ^ set(set3))
rank_e = list (set(set5) ^ set(set4))

whole = np.concatenate([rank_first, rank_a,rank_b,rank_c,rank_d,rank_e])
len(whole)

sed_epi = np.array([0.0] * NBR_ITEMS)

sed_epi[rank_first] = s[rank_first]
sed_epi[rank_a] = s1[20:25]
sed_epi[rank_b] = s2[25:45] 
sed_epi[rank_c] = s3[45:65]
sed_epi[rank_d] = s4[65:85]
sed_epi[rank_e] = s5[85:100]

sed
dat['sed_epi'] = sed_epi
dat['bcr_epis'] = sed_epi / cost
topname_epis = ["Top_epis" + str(i) for i in top]
topname_epis
for t in range(len(top)):
    dat[topname_epis[t]] = [0] * NBR_ITEMS
    dat.loc[dat.nlargest(
        top[t], 'bcr_epis').index, topname_epis[t]] = 1

sedsum_epis = [sum(sed_epi * dat[i]) for i in topname_epis]
sedsum_epis
costsum_epis = [sum(cost * dat[i]) for i in topname_epis]
costsum_epis
dat.columns
sedsum_ignore_epis = [sum(sed_epi * dat[i]) for i in topname]
sedsum_ignore_epis
costsum_ignore_epis = [sum(cost * dat[i]) for i in topname]
costsum_ignore_epis

plt.scatter(sedsum_ignore_epis, costsum_ignore_epis,
            c='b', marker='o', label='ignore_epistas_bcr')
plt.scatter(sedsum_epis, costsum_epis, c='c',
            marker='o', label='epistasis_bcr')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')

############################
##EA

dat['ID'].iloc[[74,3,8,2,0]]





#Gamma distribution
shape, scale = 1, 2.
plt.subplots(figsize=(10,5))
np.random.seed(seed=32)
s_gamma = np.random.gamma(shape, scale, 50000)
plot=sns.distplot(pd.DataFrame(s_gamma)).set_title('Gamma Distribution with $shape$=%.1f and $scale$=%.1f'%(shape,scale))
fig=plot.get_figure()

s_gamma1=np.random.gamma(5,scale,5000)
np.random.seed(seed=32)
fig,(ax1,ax2)= plt.subplots(1,2, figsize=(15, 4))
 
plot=sns.distplot(pd.DataFrame(s_gamma),ax=ax1).set_title('Gamma Distribution with shape parameter =1')
fig=plot.get_figure()
 
plot1=sns.distplot(pd.DataFrame(s_gamma1),ax=ax2).set_title('Gamma Distribution with shape parameter =5')
fig1=plot1.get_figure()


s_gamma2=np.random.gamma(shape,5,5000)
np.random.seed(seed=32)
fig, (ax1,ax2) =plt.subplots(1,2,figsize=(15,4))
 
plot=sns.distplot(pd.DataFrame(s_gamma),ax=ax1).set_title('Gamma Distribution with scale parameter=2')
fig=plot.get_figure()
 
plot1=sns.distplot(pd.DataFrame(s_gamma2),ax=ax2).set_title('Gamma Distribution with scale parameter=5')
fig1=plot1.get_figure()