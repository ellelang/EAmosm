
from pathlib import Path
import numpy as np
import pandas as pd
import wcmofunctions as fun 
mdata_folder = Path("C:/Users/langzx/Documents")
wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
WCMO_N = wcmo.shape[0]
wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
flowdata = pd.read_csv(mdata_folder/"SWAT_WY.csv")
SBIO = pd.read_csv(mdata_folder/"SB IO Detail.csv")


WaterYeild = flowdata['Water Yield']
subbasin = flowdata['Subbasin']

SBarea_m2 = SBIO['SBarea_m2']
sigma = 0.7
n = 30 * 9131
Headerwater = SBIO['Headwater']
upSBs = SBIO['up_SBs']
ID_next = SBIO['ID_next']
C1 = SBIO['C1']
C2 = SBIO['C2']
C3 = SBIO['C3']


WCMOselect =  fun.allocate (wcmo,30,WCMO_N)
SB_mocount = WCMOselect [0]
SB_mocount
SB_moaream2 = WCMOselect [1]
Qtnout_record = fun.riverrout(WaterYeild, SBarea_m2,SB_mocount,SB_moaream2)
QtnoutriverMCrouted = fun.apply_riverrout (sigma,n,Qtnout_record, Headerwater,upSBs, ID_next,C1,C2,C3)
outputdata =  pd.DataFrame({'Subbasin':subbasin, 'QtnoutriverMCrouted':QtnoutriverMCrouted})
outputdata.to_csv(mdata_folder/"output.csv", index=False)

sd = fun.hockystick (outputdata,4, 13, 19, 2010-1985)