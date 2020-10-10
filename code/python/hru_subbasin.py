from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
#data_folder = Path('/Users/zhenlang/Desktop/github/EAmosm/data')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import array
import random
import json

hru_id_area = pd.read_csv (data_folder/'MRB2020/HRU_ids_areas.csv')
hru_lookup = pd.read_csv (data_folder/'MRB2020/HRU_Lookup.csv')

hru_merge = pd.merge(hru_lookup, hru_id_area,how='left',left_on='HRU Number', right_on='HRU_ID')

hru_merge.head(3)

hru_merge.to_csv(data_folder/'MRB2020/hru_merge.csv',index = False )
hru_merge.columns

genome = hru_merge['Genome Position']
gen_c = np.repeat(genome, 15)
# gen_c
# len(gen_c)
# gen_c
# gen_c.to_csv(data_folder/'MRB2020/geno.csv')

hru_label = pd.read_csv (data_folder/'MRB2020/hru_labels.csv')
hru_label.shape
#hru_label['number_label'] = 
hru_label.columns
label_merge = pd.merge(hru_label, hru_merge, how = 'left', left_on = 'Genome Position', right_on = 'Genome Position') 
label_merge.to_csv(data_folder/'MRB2020/hrulabel_merge.csv', index = False)
######################

sub_counties = pd.read_csv(data_folder/"MRB2020/subbasincounties.csv")
temp_table1 = sub_counties.groupby(["Subbasin"])['County'].apply(list)
temp_table1

sub_id = pd.read_csv(data_folder/"MRB2020/subbasin_ids.csv")
sub_id.head(3)
practice = sub_id.Practices.unique().tolist()
practice
sub_new = sub_id.groupby("subbasin")["Select"].apply(list).reset_index(name='new')
sub_new.columns
sub_new.head(3)

sub_result = pd.DataFrame(sub_new['new'].to_list(), columns=practice)
sub_result['subbasin'] = sub_new['subbasin'].str.extract('(\d+)').astype(int)
sub_result.head(3)
sub_result.to_csv(data_folder/"MRB2020/subbasin_result.csv", index = False)
################


hru_idlabel = pd.read_csv(data_folder/"MRB2020/hru_idlabels.csv")
hru_idlabel.columns
hru_practice = hru_idlabel.Practice.unique().tolist()
hru_practice
len(hru_idlabel.HRU.unique().tolist())
hru_new = hru_idlabel.groupby("HRU")["Select"].apply(list).reset_index(name='new')
hru_new.columns
hru_new.head(3)
hru_result = pd.DataFrame(hru_new['new'].to_list(), columns=hru_practice)
hru_result
hru_result['HRU'] = hru_new['HRU']
hru_result.to_csv(data_folder/"MRB2020/hru_result.csv", index = False)
