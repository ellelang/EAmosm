from pathlib import Path
#data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
data_folder = Path('/Users/ellelang/Desktop/github/EAmosm/data')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import array
import random
import json
from pprint import pprint

data = pd.read_csv (data_folder/'LS2021/1AAT_clean.csv')
data.columns
data_sub = data[data.Genome == 'sub'].reset_index()
df_ld05 = data_sub[data_sub['lambda0.5'] > 0].sort_values('lambda0.5',ascending=False).reset_index()
df_ld05.head()

data.shape

agglist = np.array([1, 2, 3, 4, 5, 6])
def statReport(list_rep):
    mean = np.mean(list_rep)
    std = np.std(list_rep)
    
    q_25 = np.quantile(list_rep, 0.25)
    q_50 = np.median(list_rep)
    q_75 = np.quantile(list_rep, 0.75)
    report_list = [mean, std, q_25, q_50, q_75]
    return report_list

statReport(agglist)



def ratio_div (df, ldsce):
    df_clean = df[df[ldsce]>0].reset_index()
    ld_list = df_clean[ldsce]
    ratio_dict = dict()
    for i in range(len(ld_list)):
        ratio = np.divide(ld_list[i], ld_list.tolist())
        ratio_dict[df_clean['Names'][i]] = statReport(ratio)
    return ratio_dict

ld = np.round(np.arange(0,1.1,0.1),1)
ldname = ["lambda" + str(i) for i in ld]
ldname

df_bcr_list = []
for i in ldname:
    ld_dist = ratio_div(data, i)
    df = pd.DataFrame(list(ld_dist.items()),columns = ['Names','Stats'])
    df_report = pd.concat([df['Names'], df['Stats'].apply(pd.Series)], axis = 1)
    df_report.columns = ['Name', 'mean', 'std', 'q25', 'q50', 'q75']
    df_report['ld'] = i
    df_bcr_list.append(df_report)


ratio_comb = pd.concat(df_bcr_list, axis = 0)
ratio_comb.to_csv(data_folder/'LS2021/ratio_distribution.csv', index = False)
###################
ratio_all = pd.read_csv(data_folder/'LS2021/ratio_distribution.csv')
ratio_all.shape
ratio_all.head(4)

ratio_all['Name_str'] = ratio_all['Name'].str.replace(r'\d+','').replace(r'\D', '')
ratio_all['Name_str'].unique()

ratio_all.loc[(ratio_all.Name_str == 'RAMO-'), 'Name_str'] = 'RAMO'
ratio_all.loc[(ratio_all.Name_str == 'NCMO-'), 'Name_str'] = 'NCMO'
ratio_all['Name_str'].unique()


ratio_mean_ld = ratio_all.groupby(['ld', 'Name_str'])['mean', 'std', 'q25', 'q50', 'q75'].mean().reset_index()

ratio_mean_ld.to_csv(data_folder/'LS2021/ratio_distribution_ld.csv', index = False)

ratio_mean_all = ratio_all.groupby(['Name_str'])['mean', 'std', 'q25', 'q50', 'q75'].mean().reset_index()

ratio_mean_all.to_csv(data_folder/'LS2021/ratio_distribution_all.csv', index = False)




ld05_dist = ratio_div(data_sub,'lambda0.5')
df = pd.DataFrame(list(ld05_dist.items()),columns = ['Names','Stats'])
df_report = pd.concat([df['Names'], df['Stats'].apply(pd.Series)], axis = 1)
df_report.head()
df_report.columns = ['Name', 'mean', 'std', 'q25', 'q50', 'q75']
df_report.to_csv(data_folder/'LS2021/subld05_distribution.csv')


sub_ratio = pd.concat(df_bcr_list, axis = 0)
sub_ratio.to_csv(data_folder/'LS2021/ratio_sub.csv', index = False) 


ratio_list = []
for i in ldname:
    ldratio = ratio_div(data_sub,i)
    ratio_list.append(ldratio)

len(ratio_list)




df_ld05.groupby('Names_str')['bcr_dist'].mean().sort_values(ascending=False).reset_index()

df_ld05.groupby('Names_str')['bcr_dist'].mean().rank(ascending = False)

