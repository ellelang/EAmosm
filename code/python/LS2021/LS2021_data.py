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

data_1aat = pd.read_csv (data_folder/'LS2021/ldscedf0803ALL.csv')
nsize = len (data.index)
nsize



hru_data = data_1aat[data_1aat.Genome == 'hru']
hru_data['Names_str'] = hru_data['Names'].str.replace(r'\d+','')
hru_data['Names_str']
hru_data['Names_int'] = hru_data.Names.str.extract('(\d+)').astype(int)


sub_data = data_1aat[data_1aat.Genome == 'sub']
sub_data['Names']
sub_data['Names_str'] = sub_data['Names'].str[7:]
sub_data['Names_str']
sub_data['Names_int'] = sub_data.Names.str.extract('(\d+)').astype(int)
sub_data['Names_int'] 
data_hru_sub = pd.concat([sub_data, hru_data], axis = 0)
data_hru_sub.to_csv(data_folder/'LS2021/sub_hru.csv', index = False)
