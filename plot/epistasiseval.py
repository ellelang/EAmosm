from pathlib import Path
data_folder = Path("C:/Users/langzx/OneDrive/AAMOSM2018/ECR/Rplotresults")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns
import SeabornFig2Grid as sfg

wbcreval = pd.read_csv(data_folder/'WBCReval.csv')
wbclinear = pd.read_csv(data_folder/'WBCRlinear.csv')
scaleratio = pd.read_csv(data_folder/'scalratiodiff.csv')


scaleratio.head()
sns.set_palette("pastel")
pd1 = pd.concat([wbcreval, wbclinear])
pd1['Cost($Million/Year)'] = pd1['Cost']/1000000
pd1.head()


g1 = sns.lmplot(x="Cost($Million/Year)", y="Sediment.Reduction", hue="EA frontier", truncate=True, height=5, data=pd1,fit_reg=False,legend_out=False)
g1.savefig("diff.pdf")

sns.set_palette("Set3")
g2 = sns.lmplot(x="Sediment.Reduction.Share", y="Scaling.Ratio", order=2,data=scaleratio, fit_reg=True,  truncate=True)
g2.fig.text(0.40, 0.905,'$y = -0.5818x^2 - 0.0465x + 0.9896$', fontsize=12)

fig = plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(1,2)
mg0 = sfg.SeabornFig2Grid(g1, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g2, fig, gs[1])

gs.tight_layout(fig)
fig.text(0.65, 0.915,'$y = -0.5818x^2 - 0.0465x + 0.9896$', fontsize=12)
#gs.update(top=0.7)

plt.show()
fig.savefig('diffratio.pdf')   
