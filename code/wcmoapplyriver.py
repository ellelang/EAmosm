
from pathlib import Path
import numpy as np
import pandas as pd
mdata_folder = Path("C:/Users/langzx/Documents")

wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
flowdata = pd.read_csv(mdata_folder/"SWAT_WY.csv")
SBIO = pd.read_csv(mdata_folder/"SB IO Detail.csv")
SBoutput = pd.read_csv(mdata_folder/"Output Subbasin Daily.csv")
sigma = 0.7
count = np.empty (30)
n = 30 * 9131
Qtnoutriver = np.empty (n)
QtnoutriverMCrouted = np.empty (n)
Qtnout =np.random.uniform(1,10,n)
Qtnout
SBIO['Headwater']
upSBs  =  SBIO['up_SBs']
#######HEADWATER: move down river starting from upppermost SB
for m in range(29, -1, -1):
    if SBIO['Headwater'][m] == "Y":
        i = m
        count [i] = count [i] + 1
        # call storageoutflow fucntion to generate Qtnout (index)
        for j in range (0,9131):
            index  = j + ((i + 1) - 1) * 9131
            Qtnoutriver [index] = Qtnout [index]
        # initial MCrouted values
        index_initial = 0 + ((i + 1) - 1) * 9131
        QtnoutriverMCrouted [index_initial] = sigma * Qtnout [index_initial]
        
        # Muskingum routing
        for j in range (1,9131):
            index_2 = j + ((i + 1) - 1) * 9131
            QtnoutriverMCrouted [index_2] = SBIO['C1'][i] * Qtnoutriver [index_2] \
                                          + SBIO['C2'][i] * Qtnoutriver [index_2 - 1] \
                                          + SBIO['C3'][i] * QtnoutriverMCrouted [index_2 - 1]
        ##########END HEADWATER
        ###MOVE DOWN SB UNTIL REACHING THE TERMINAL
        o = i
        i = SBIO['ID_next'][o]
        while True:
            count [i] = count [i] + 1
            if count [i] == 1:
                Qtnout = np.random.uniform(1,10,n)
            
            for j in range (0,9131):
                index = j + ((i + 1) - 1) * 9131
                index_o = j + ((o + 1) - 1) * 9131
                QtnoutriverMCrouted [index] = Qtnoutriver [index] \
                                          + Qtnout[index] - (count[i] - 1) * Qtnout[index] \
                                          + QtnoutriverMCrouted [index_o]
            upSBs [i] = upSBs [i] - 1
            if upSBs [i] != 0:
                break
            index_initial = 0 + ((i + 1) - 1) * 9131
            QtnoutriverMCrouted [index_initial] = sigma * Qtnoutriver [index_initial]
            
            for j in range (1,9131):
                index_2 = j + ((i + 1) - 1) * 9131
                QtnoutriverMCrouted [index_2] = SBIO['C1'][i] * Qtnoutriver [index_2] \
                                          + SBIO['C2'][i] * Qtnoutriver [index_2 - 1] \
                                          + SBIO['C3'][i] * QtnoutriverMCrouted [index_2 - 1]
            
            o = i
            ii = SBIO['ID_next'][i]
            i = ii
            if i == 0:
                break


## generate flow output QtnoutriverMCrouted
                