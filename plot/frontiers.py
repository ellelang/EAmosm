from pathlib import Path
#data_folder = Path("C:/Users/langzx/OneDrive/AAMOSM2018/ECR/Rplotresults")
data_folder = Path('D:/OneDrive/AAMOSM2018/ECR/Rplotresults')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=14)     
plt.rc('ytick', labelsize=14)
import seaborn as sns
import pyreadr
 

ewbcrFront = pyreadr.read_r('D:/OneDrive/AAMOSM2018/ECR/Rplotresults/Evaluated_wBCR.rds')
noseedsFront = pyreadr.read_r ('D:/OneDrive/AAMOSM2018/ECR/Rplotresults/noseeds.rds')
uniformseed099Front = pyreadr.read_r ('D:/OneDrive/AAMOSM2018/ECR/Rplotresults/uniformseed099.rds')
bcr_uniform099Front = pyreadr.read_r ('D:/OneDrive/AAMOSM2018/ECR/Rplotresults/04201610bcr_uniform099.rds')
##########convert RDS to dataframe
dfbcr_eval = ewbcrFront[None]
dfnoseed = noseedsFront[None]
dfuniform = uniformseed099Front[None]
dfbcr_uniform= bcr_uniform099Front[None]

colormap = ['seagreen','steelblue','slateblue','goldenrod']

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.8, hspace=None)
fig, [ax1,ax2] = plt.subplots(1,2, figsize=(20,12))

ax1.scatter(dfbcr_uniform['Outlet.Sediment'], dfbcr_uniform['Cost']/1000000, label=r'$\mathcal{F}_{unif+wBCR}$',  color =colormap[0])
ax1.scatter(dfuniform['Outlet.Sediment'], dfuniform['Cost']/1000000,label=r'$\mathcal{F}_{unif+wBCR}$',color =colormap[1])
ax1.scatter(dfnoseed['Outlet.Sediment'], dfnoseed['Cost']/1000000,label=r'$\mathcal{F}_0$',color =colormap[2])
ax1.scatter(dfbcr_eval['Outlet.Sediment'], dfbcr_eval['Cost']/1000000,label=r'$\mathcal{F}_{evaluated\, wBCR}$',color =colormap[3])
ax1.grid(False)
ax1.legend(fontsize=18)
ax1.set_xlabel("(a).Outlet sediment (mg/yr)",fontsize=18)
ax1.set_ylabel("Cost ($ Million /Year)", fontsize=18)
ax1.set_ylim(0,160)


ax2.scatter(dfbcr_uniform['Duck'], dfbcr_uniform['Cost']/1000000, color =colormap[0])
ax2.scatter(dfuniform['Duck'], dfuniform['Cost']/1000000,color =colormap[1])
ax2.scatter(dfnoseed['Duck'], dfnoseed['Cost']/1000000,color =colormap[2])
ax2.scatter(dfbcr_eval['Duck'], dfbcr_eval['Cost']/1000000,color =colormap[3])

ax2.grid(False)
ax2.set_xlabel("(b).Annual duck hatchlings",fontsize=18)
ax2.set_ylabel("Cost ($ Million /Year)",fontsize=18)
ax2.set_ylim(0,160)

fig.show()
plt.savefig('frontiers.png',bbox_inches='tight', dpi=500)
