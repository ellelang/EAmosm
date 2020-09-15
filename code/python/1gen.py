from pathlib import Path
#data_folder = Path("C:/Users/langzx/Desktop/github/EAmosm/data")
data_folder = Path('/Users/zhenlang/Desktop/github/EAmosm/data')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import array
import random
import json
import altair as alt

dat = pd.read_csv (data_folder/'MRB2020/MRB1gen0909.csv')
dat.columns
baseline_nit = dat['Reach Outlet NO3 kg/yr'][0]
baseline_sed = dat['Reach Outlet Sediment tons/yr'][0]

nitRed = baseline_nit - dat['Reach Outlet NO3 kg/yr']
sedRed = baseline_sed - dat['Reach Outlet Sediment tons/yr']
cost = dat['Cost']

bcr_nit = nitRed/cost
bcr_sed = sedRed/cost

dat['bcr_nit'] = bcr_nit
dat['bcr_sed'] = bcr_sed

dat['nitRed'] = nitRed
dat['sedRed'] = sedRed

dat_new = dat.iloc[1:]

dat_new['bcr_nit']

name_by_nit = dat_new.sort_values("bcr_nit",  ascending=False)['Name'].tolist()

name_by_sed = dat_new.sort_values("bcr_sed",  ascending=False)['Name'].tolist()

name_by_sed 

d = {'bcr_NO3':name_by_nit,'bcr_sed':name_by_sed}
df = pd.DataFrame(d)
df.to_csv(data_folder/'1gen_bcr_name.csv', index=False, header=True)

alt.Chart(dat_new).mark_bar().encode(
    x='Name',
    y='bcr_nit'
)

fig = plt.figure()

f, axs = plt.subplots(1,2,figsize=(15,8))

axs[0].bar(dat_new['Name'], dat_new['bcr_nit'], color = 'b', width = 0.5)
axs[0].tick_params(labelrotation=90)
axs[0].set_title("BCR NO3")
axs[0].set_ylim(0, None)
axs[1].bar(dat_new['Name'], dat_new['bcr_sed'], color = 'r', width = 0.5)
axs[1].tick_params(labelrotation=90)
axs[1].set_title("BCR Sediment")
axs[1].set_ylim(0, None)



fig = plt.figure()

f, axs = plt.subplots(1,3,figsize=(25,8))

axs[0].bar(dat_new['Name'], dat_new['nitRed'], color = 'b', width = 0.5)
axs[0].tick_params(labelrotation=90)
axs[0].set_title("NO3 Reduction")
axs[0].set_ylim(0, None)
axs[1].bar(dat_new['Name'], dat_new['sedRed'], color = 'r', width = 0.5)
axs[1].tick_params(labelrotation=90)
axs[1].set_title("Sediment Reduction")
axs[1].set_ylim(0, None)
axs[2].bar(dat_new['Name'], dat_new['Cost'], color = 'g', width = 0.5)
axs[2].tick_params(labelrotation=90)
axs[2].set_title("Cost")
axs[2].set_ylim(0, None)


x = np.arange(100)
source = pd.DataFrame({
  'x': x,
  'f(x)': np.sin(x / 5)
})

alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
)
