##oop example
from pathlib import Path
import numpy as np
import pandas as pd
mdata_folder = Path("C:/Users/langzx/Documents")


wcmo = pd.read_excel(mdata_folder/"WCMO.xlsx")
flowdata = pd.read_excel(mdata_folder/"SWAT_WY.xlsx")
print(flowdata['Subbasin'])
SBIO = pd.read_excel(mdata_folder/"SB IO Detail.xlsx")
SBoutput = pd.read_excel(mdata_folder/"Output Subbasin Daily.xlsx")


'''
 Section 1: Allocate WCMO to each HYDSB
 Section 2: Call water yield data
 Section 3: River routing (MG-CG) --> call sub storage outflow
 Section 4: Hockey stick application

'''
flowdata.loc[flowdata['Subbasin'] == 1]['Water Yield'][0:3]
WCMO_N = wcmo.shape[0]
wcmo['select'] = np.random.randint(2, size=WCMO_N)


# Section 1: Allocate WCMO to each HYDSB
def allocate (mo, sb_n, mo_n):
  SB_mocount = np.zeros(sb_n)
  SB_moaream2 = np.zeros(sb_n)
  for i in range (0,sb_n):
      for j in range (0,mo_n):
          if mo['HYDSB_LES30SB'][j] == i and mo['select'][j] == 1:
             SB_mocount [i] =  SB_mocount [i] + 1
             SB_moaream2 [i] = SB_moaream2 [i] + mo['area_m2'][j]
  return SB_mocount,SB_moaream2

WCMOselect =  allocate (wcmo,30,WCMO_N)

# Section 2: Call water yield data

# Section 3: River routing (MG-CG) 
## Define parameters

C = 2
Anmin_factor = 5/100
alpha = 0.5
k = 1e-7
ET = 1.16e-8
Anmin = SBIO['SBarea_m2'] * Anmin_factor 
WCMO_D = 6.6 * 0.3048
WCMO_DAfactor = 8.9
dt = 86400
n = 30 * 9131
n
SB_mocount = WCMOselect [0]
SB_moaream2 = WCMOselect [1]
len(SB_mocount )
### If no WCMO is selected:
SB_mocount_zero =  30 - (np.count_nonzero(SB_mocount))
SB_mocount_zero
Qnout = np.empty (SB_mocount_zero)
n_zero = 0
for i in range (0, 30):
  if SB_mocount[i] == 0:
      n_zero = n_zero + 1
      for j in range (0, 9131):
          index = j + (n_zero - 1) * 9131
          #Qnout[index] = flowdata.loc[flowdata['Subbasin'] == i]['Water Yield'][j]
          print (index)

aa  = 6 * 9131
aa
### When there are WCMO selected
Aw = np.empty(30)
Awbar = np.empty (30)
WCMO_SAbar = np.empty (30)
It = np.empty (n)
St = np.empty (n)
Ht = np.empty (n)
Qt = np.empty (n)
Qtnout = np.empty (n)
Awbar [1] / SBIO['SBarea_m2'][1]
flowdata = flowdata.loc[flowdata["Subbasin"] == 1]
a = flowdata['Water Yield']
len(a)
for i in range (0,30):
    if WCMO_DAfactor * SB_moaream2 [i] > SBIO['SBarea_m2'][i] - Anmin [i]:
        Aw [i] = SBIO['SBarea_m2'][i] - Anmin [i]
    else:
        Aw [i] = WCMO_DAfactor * SB_moaream2 [i]
    Awbar [i] = Aw [i] / SB_mocount [i]
    WCMO_SAbar [i] =  SB_moaream2 [i] / SB_mocount [i]      
    circ = np.sqrt(WCMO_SAbar [i]/np.pi) * 2 * np.pi
    L = alpha * circ
    
    ##Set initial condition of water
    flowdata = flowdata.loc[flowdata['Subbasin'] == i]
    It [0:9130] = flowdata['Water Yield'][0:9130] * (Awbar [i] / SBIO['SBarea_m2'][i])
    St [0:9130] = 0
    Ht [0:9130] = St [0:9130] / WCMO_SAbar [i] 
    Qt [0:9130] = 0
    ### iteration begins:
    for j in range (1,9131):
        It [9130 + j] = flowdata['Water Yield'][j] * (Awbar [i] / SBIO['SBarea_m2'][i])
        #St [j] = St [j - 1] + (It [j - 1] - Qt [j - 1] - WCMO_SAbar [i] ) * (1 * k + ET)) * dt 
        St [9130 + j] = 1
        Ht [9130 + j] = St [9130 + j] / WCMO_SAbar [i] 
        if Ht [9130 + j] > WCMO_D:
            Qt [9130 + j] = C * L * (Ht [9131 + 0] - WCMO_D) ^ 1.5
        else:
            Qt [9131 + 0] = 0
    for j in range (0,9131):
        Itn [j] = It [j] * SB_mocount [i] 
        Stn [j] = St [j] * SB_mocount [i] 
        Htn [j] = Ht [j] * SB_mocount [i] 
        Qtn [j] = Qt [j] * SB_mocount [i]
        Qtnout [j] = Qtn [j] + (flowdata['Water Yield'][j] - Itn [j] )
        flowdata['Water Yield'][j] = Qtnout [j]