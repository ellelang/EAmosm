#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:30:59 2022

@author: ellelang
"""

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

data_1aat = pd.read_excel (data_folder/'MRB2022/1AAT20221014.xlsx',
                           sheet_name="Results")

data_1aat.shape


df_name = data_1aat[['Seed Label']]
df_name.iloc[0]

def str_genome (df):
    label = df['Seed Label'].tolist()
    geno_lst = []
    for i in label:
        if 'Sub' in i:
            genoname_i = 'sub'
        elif 'RAMO' in i:
            genoname_i = 'ravine'
        elif 'NCMO' in i:
            genoname_i = 'nearChannel'
        elif i == "Nothing":
            genoname_i = 'Nothing'
        else:
            genoname_i = 'hru'
        geno_lst.append(genoname_i)
        
    df['Genome'] = geno_lst
    df['Label'] = df['Seed Label']
    
    return df


df_name_all = str_genome (df_name)


df_name_all.query('Genome == "sub"')


hru_data = df_name_all[df_name_all.Genome == 'hru']
hru_data[['Practice', 'subID']] = hru_data['Label'].str.split(" ", 1, expand = True)
hru_data.head()
hru_data['Genome section'] = 'HRU'

hru_data['WithinSectionID'] = hru_data.Practice.str.extract('(\d+)').astype(int)
hru_data['AlleleCombo'] = hru_data.Practice.str.replace(r'\d+','')
##############

# hru_data[['subbasin', 'HRU']] = hru_data['subID'].str.split("-",  expand = True)
# hru_data.head()
hru_df = hru_data[['Label', 'Genome section','WithinSectionID', 'AlleleCombo']]



sub_data = df_name_all[df_name_all.Genome == 'sub']
sub_data['Practice'] = sub_data['Label'].str[7:]

sub_data['WithinSectionID'] = sub_data.Label.str.extract('(\d+)').astype(int)
sub_data['AlleleCombo'] = sub_data['Label'].str[7:]
sub_data['Genome section'] = 'sub' + sub_data['Label'].str[7:10]

sub_data['Genome section'] 
sub_df = sub_data[['Label', 'Genome section', 'WithinSectionID', 'AlleleCombo']]
###################3
hru_others = df_name_all[(df_name_all.Genome == 'ravine')|(df_name_all.Genome == 'nealChannel')]
hru_others['WithinSectionID'] = hru_others.Label.str.extract('(\d+)').astype(int)
hru_others['AlleleCombo'] = hru_others['Label'].str.replace(r'\d+','')
hru_others.head()

hru_others['Genome section'] = hru_others['AlleleCombo']
# hru_others[['subbasin', 'HRU']] = 0
# hru_others['Practice'] = hru_others['Names'] 
others_df = hru_others[['Label', 'Genome section', 'WithinSectionID', 'AlleleCombo']]

################
df_nothing = df_name_all[df_name_all.Genome == 'Nothing']
df_nothing['WithinSectionID'] = 0
df_nothing['AlleleCombo'] = "Nothing"
df_nothing['Genome section'] = df_nothing['AlleleCombo']
df_nothing = df_nothing[['Label', 'Genome section', 'WithinSectionID', 'AlleleCombo']]


#############
data_hru_sub = pd.concat([df_nothing, sub_df, hru_df, others_df], axis = 0, ignore_index = True)
data_hru_sub.shape
data_hru_sub.to_csv(data_folder/'MRB2022/genolabels.csv', index = False)

######1AAT results
df_1014 = pd.read_csv(data_folder/'MRB2022/1AAT20221014.csv')
df_1014.shape
df_1014.Label


df_all = pd.merge(df_1014, data_hru_sub, on = 'Label')
df_all.to_csv(data_folder/'MRB2022/1AAT1014compile.csv', index = False)



########

df_all['AlleleCombo'].unique()

