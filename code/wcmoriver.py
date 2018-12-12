##oop example
from pathlib import Path
import numpy as np
import pandas as pd
mdata_folder = Path("C:/Users/langzx/Documents")


wcmo = pd.read_excel(mdata_folder/"WCMO.xlsx")
flowdata = pd.read_excel(mdata_folder/"SWAT_WY.xlsx")
SBIO = pd.read_excel(mdata_folder/"SB IO Detail.xlsx")
SBoutput = pd.read_excel(mdata_folder/"Output Subbasin Daily.xlsx")


'''
 Section 1: Allocate WCMO to each HYDSB
 Section 2: Call water yield data
 Section 3: River routing (MG-CG) --> call sub storage outflow
 Section 4: Hockey stick application

'''

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
  return SB_mocount,SB_aream2

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

### If no WCMO is selected:
Qnout = np.zeros (n)
for i in range (0, 30):
  if SB_mocount[i] == 0:
      for j in range (0, 9131):
          Qnout[j] = flowdata.loc[flowdata['Subbasin'] == i]['Water Yeild'][j]
      
### When there are WCMO selected
Aw = np.empty(30)
Awbar = np.empty (30)
WCMO_SAbar = np.empty (30)
It = np.empty (n)
St = np.empty (n)
Ht = np.empty (n)
Qt = np.empty (n)
Qtnout = np.empty (n)
for i in range (0,30):
    if WCMO_DAfactor * SB_moaream2 [i] > SBIO['SBarea_m2'] - Anmin:
        Aw [i] = SBIO['SBarea_m2'] - Anmin
    else:
        Aw [i] = WCMO_DAfactor * SB_moaream2 [i]
    Awbar [i] = Aw [i] / SB_mocount [i]
    WCMO_SAbar [i] =  SB_moaream2 [i] / SB_mocount [i]      
    circ = np.sqrt(WCMO_SAbar [i]/np.pi) * 2 * np.pi
    L = alpha * circ
    
    ##Set initial condition of water
    flowdata = flowdata.loc[flowdata['Subbasin'] == i]
    It [1] = flowdata['Water Yeild'][i] * (Awbar [i] / SBIO['SBarea_m2'][i])
    St [1] = 0
    Ht [1] = St [1] / WCMO_SAbar [i] 
    Qt [1] = 0
    ### iteration begins:
    for j in range (1,9131):
        It [j] = flowdata['Water Yeild'][j] * (Awbar [i] / SBIO['SBarea_m2'][i])
        #St [j] = St [j - 1] + (It [j - 1] - Qt [j - 1] - WCMO_SAbar [i] ) * (1 * k + ET)) * dt 
        St [j] = 1
        Ht [j] = St [j] / WCMO_SAbar [i] 
        if Ht [j] > WCMO_D:
            Qt [j] = C * L * (Ht [j] - WCMO_D) ^ 1.5
        else:
            Qt [j] = 0
    for j in range (0,9131):
        Itn [j] = It [j] * SB_mocount [i] 
        Stn [j] = St [j] * SB_mocount [i] 
        Htn [j] = Ht [j] * SB_mocount [i] 
        Qtn [j] = Qt [j] * SB_mocount [i]
        Qtnout [j] = Qtn [j] + (flowdata['Water Yeild'][j] - Itn [j] )
        flowdata['Water Yeild'][j] = Qtnout [j]