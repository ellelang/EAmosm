import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import scipy.special as sps
from pathlib import Path
import seaborn as sns
import itertools

#data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data/output')
#icmo_maple = pd.read_csv (data_folder/"icmo_MAP_sub232119.csv")
#wcmo_maple23 = pd.read_csv (data_folder/"wcmo_MAP_sub23.csv")
#wcmo_maple21 = pd.read_csv (data_folder/"wcmo_MAP_sub21.csv")
#wcmo_maple19 = pd.read_csv (data_folder/"wcmo_MAP_sub19.csv")
#icmo_maple.columns
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data/csv')
wcmoga = pd.read_csv (data_folder/"wcmogawhole.csv")
icmomaple = pd.read_csv (data_folder/"icmo_MAP_sub232119.csv")
wcmo_subset = wcmoga[ (wcmoga['HYDSB_LES30SB']==23) | (wcmoga['HYDSB_LES30SB']==21 )| (wcmoga['HYDSB_LES30SB']==19)]
wcmoSedRed = wcmo_subset.groupby(['HYDSB_LES30SB'])['SedRed'].sum().reset_index()
icmoSedRed = icmomaple.groupby(['Subbasin_2'])['SedRed'].sum().reset_index()

wcmoSedRed.columns = ['Subbasin','SedRed']
icmoSedRed.columns = ['Subbasin','SedRed']
SedRedSum = pd.merge([wcmoSedRed, icmoSedRed])
SedSum = pd.concat([wcmoSedRed,icmoSedRed]).groupby(['Subbasin'])['SedRed'].sum().reset_index()
pmt = itertools.product([0,1], repeat=3)
pmtmatrix = np.matrix(list(pmt))
sumSed = pmtmatrix.dot(SedSum['SedRed'])



mdata_folder = Path("C:/Users/langzx/Documents")
wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
wcmo_subset = wcmo[ (wcmo['HYDSB_LES30SB']==23) | (wcmo['HYDSB_LES30SB']==21 )| (wcmo['HYDSB_LES30SB']==19)]
wcmo_subset.shape
wcmo_subset['select'] = np.repeat(1,405)

flowdata = pd.read_csv(mdata_folder/"SWAT_WY.csv")
flowdata_subset = flowdata[(flowdata['Subbasin'] == 23)|(flowdata['Subbasin'] == 21)|(flowdata['Subbasin'] == 19)]
SBIO = pd.read_csv(mdata_folder/"SB IO Detail.csv")
SBIO_subset = SBIO[(SBIO['ID'] == 23)|(SBIO['ID'] == 21)|(SBIO['ID'] == 19)]

subbasin = flowdata_subset['Subbasin'].reset_index()['Subbasin']


sub_select = np.array([19,21,23])
SB_moaream2 = wcmo_subset.groupby(['HYDSB_LES30SB'])['area_m2'].sum().reset_index()['area_m2']
SB_mocount = wcmo_subset.groupby(['HYDSB_LES30SB']).size().reset_index()['HYDSB_LES30SB']
SB_mocount[0]

WaterYeild = flowdata_subset['Water Yield'].reset_index()['Water Yield']
WaterYeild.shape
# Section 3: River routing (MG-CG)

SBarea_m2 = SBIO_subset['SBarea_m2'].reset_index()['SBarea_m2']

def riverrout (WaterYeild, SBarea_m2,SB_mocount,SB_moaream2):
    C = 2
    Anmin_factor = 5/100
    alpha = 0.5
    k = 1e-7
    ET = 1.16e-8
    Anmin= SBarea_m2 * Anmin_factor
    WCMO_D = 6.6 * 0.3048
    WCMO_DAfactor = 8.9
    dt = 86400
    n = 3 * 9131
    Aw = np.empty(3)
    Awbar = np.empty (3)
    WCMO_SAbar = np.empty (3)
    It = np.empty (n)
    St = np.empty (n)
    Ht = np.empty (n)
    Qt = np.empty (n)
    Itn = np.empty (n)
    Stn = np.empty (n)
    Htn = np.empty (n)
    Qtn = np.empty (n)
    
    Qtnout = np.empty (n)
### If no WCMO is selected:
    for i in range (0,3):
        if SB_mocount [i] == 0:
          for j in range (0, 9131):
                index = j + ((i + 1) - 1) * 9131
                Qtnout[index] = WaterYeild[index]
                #Qnout[index] = index
### If WCMO is selected:                
        else:
          
            if WCMO_DAfactor * SB_moaream2 [i] > SBarea_m2[i] - Anmin [i]:
                  Aw [i] = SBarea_m2[i] - Anmin [i]
            else:
                  Aw [i] = WCMO_DAfactor * SB_moaream2 [i]

            Awbar [i] = Aw [i] / SB_mocount [i]

            WCMO_SAbar [i] =  SB_moaream2 [i] / SB_mocount [i]

            circ = np.sqrt(WCMO_SAbar [i]/np.pi) * 2 * np.pi

            L = alpha * circ

              ##Set initial condition of water
            index_initial = 0 + ((i + 1) - 1) * 9131

            It [index_initial] = WaterYeild[index_initial] * (Awbar [i] / SBarea_m2[i])

            St [index_initial] = 0

            Ht [index_initial] = St [index_initial] / WCMO_SAbar [i]

            Qt [index_initial] = 0

              ### iteration begins:
            for j in range (1,9131):

                  index_2 = j + ((i + 1) - 1) * 9131

                  It [index_2] = WaterYeild[index_2] * (Awbar [i] / SBarea_m2[i])
                  St [index_2] = St [index_2 - 1] + (It [index_2 - 1] - Qt [index_2 - 1] - WCMO_SAbar [i] * (1 * k + ET)) * dt
                  Ht [index_2] = St [index_2] / WCMO_SAbar [i]
                  if Ht [index_2] > WCMO_D:
                      Qt [index_2] = C * L * (Ht [index_2] - WCMO_D) ** 1.5
                  else:
                      Qt [index_2] = 0

            for j in range (0,9131):

                  index_whole = j + ((i + 1) - 1) * 9131

                  Itn [index_whole ] = It [index_whole ] * SB_mocount [i]

                  Stn [index_whole ] = St [index_whole ] * SB_mocount [i]

                  Htn [index_whole ] = Ht [index_whole ] * SB_mocount [i]

                  Qtn [index_whole ] = Qt [index_whole ] * SB_mocount [i]

                  Qtnout [index_whole ] = Qtn [index_whole ] + (WaterYeild[index_whole ] - Itn [index_whole ] )

                  WaterYeild[index_whole] = Qtnout [index_whole]


    #Qtnout.reshape(9131,30)
    #np.where(Qtnout != 0)
    return Qtnout
Qtnout_record = riverrout(WaterYeild, SBarea_m2,SB_mocount,SB_moaream2)


# Section 3.2: Apply River_routing
 
sigma = 0.7
n = 3 * 9131
Headerwater = SBIO_subset['Headwater'].reset_index()['Headwater']
upSBs = SBIO_subset['up_SBs'].reset_index()['up_SBs']
ID_next = SBIO_subset['ID_next'].reset_index()['ID_next']
ID_next[0] = 0
ID_next.index.tolist()
C1 = SBIO_subset['C1'].reset_index()['C1']
C2 = SBIO_subset['C2'].reset_index()['C2']
C3 = SBIO_subset['C3'].reset_index()['C3']
#
#for m in range(2, -1, -1):
#    print(m)

def apply_riverrout (sigma,n,Qtnout,Headerwater,upSBs, ID_next,C1,C2,C3):
     count = np.zeros (3)
     Qtnoutriver = np.zeros (n)
     QtnoutriverMCrouted = np.zeros (n)
     #######HEADWATER: move down river starting from upppermost SB
     for m in range(2, -1, -1):
         if Headerwater[m] == "Y":
         
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
                 QtnoutriverMCrouted [index_2] = C1[i] * Qtnoutriver [index_2] \
                                               + C2[i] * Qtnoutriver [index_2 - 1] \
                                               + C3[i] * QtnoutriverMCrouted [index_2 - 1]
     #return QtnoutriverMCrouted       
           ###MOVE DOWN SB UNTIL REACHING THE TERMINAL##########END HEADWATER
             o = i
             i = ID_next.index.tolist()[o]
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
                     QtnoutriverMCrouted [index_2] = C1[i] * Qtnoutriver [index_2] \
                                               + C2[i] * Qtnoutriver [index_2 - 1] \
                                               + C3[i] * QtnoutriverMCrouted [index_2 - 1]

                 o = i
                 ii = ID_next[i]
                 i = ii
                 if i == 0:
                     break
     return QtnoutriverMCrouted

QtnoutriverMCrouted = apply_riverrout (sigma,n,Qtnout_record,Headerwater,upSBs, ID_next,C1,C2,C3)
QtnoutriverMCrouted
np.where(QtnoutriverMCrouted == 0)
outputdata =  pd.DataFrame({'Subbasin':subbasin, 'Qtnout':Qtnout_record, 'QtnoutriverMCrouted':QtnoutriverMCrouted})
outputdata.to_csv(data_folder/"routing_output.csv", index=False)



## Section 4 Calculate sediment loading from bluffs and streambanks using theHOCKEY STICK
def hockystick (outputdata,les_sb, cob_sb, map_sb,period):
    Qth = 0.01
    # Drainage area
    #LES_DA = 1156.56
    #COB_DA = 784.12
    MAP_DA = 877.15
    #Incised length
    #LES_L = 40.84
    #COB_L = 35.79
    MAP_L = 31.91

    CQ_a = 6867.9
    CQ_b = 2.1572

    #Qs_LES  = np.zeros(9131)
    #Qs_COB  = np.zeros (9131)
    Qs_MAP  = np.zeros (9131)

    #Qs_LESsum = 0
    #Qs_COBsum = 0
    Qs_MAPsum = 0

    slbaseline_MAP = 15516
    #slbaseline_COB = 17557
    #slbaseline_LES = 18771

    #SBoutput_LES = outputdata.loc[outputdata['Subbasin'] == les_sb].reset_index(inplace=False)
    #SBoutput_COB = outputdata.loc[outputdata['Subbasin'] == cob_sb].reset_index(inplace=False)
    SBoutput_MAP = outputdata.loc[outputdata['Subbasin'] == map_sb].reset_index(inplace=False)


    for j in range (0,9131):
        # populate LES/COB/MAP sediment loading from incised zone
#        if SBoutput_LES['QtnoutriverMCrouted'][j] > Qth * LES_DA:
#            Qs_LES [j] = CQ_a * (SBoutput_LES['QtnoutriverMCrouted'][j] / LES_DA)**CQ_b * LES_L
#        else:
#            Qs_LES [j] = 0

#        if SBoutput_COB['QtnoutriverMCrouted'][j] > Qth * COB_DA:
#            Qs_COB [j] = CQ_a * (SBoutput_COB['QtnoutriverMCrouted'][j] / LES_DA)**CQ_b * COB_L
#        else:
#            Qs_COB [j] = 0

        if SBoutput_MAP['QtnoutriverMCrouted'][j] > Qth * MAP_DA:
            Qs_MAP [j] = CQ_a * (SBoutput_MAP['QtnoutriverMCrouted'][j] / MAP_DA)**CQ_b * MAP_L
        else:
            Qs_MAP [j] = 0

        #Qs_LESsum = Qs_LESsum + Qs_LES [j]
        #Qs_COBsum = Qs_COBsum + Qs_COB [j]
        Qs_MAPsum = Qs_MAPsum + Qs_MAP [j]

    #slmo_LES  = Qs_LESsum / period
    #slmo_COB  = Qs_COBsum / period
    slmo_MAP  = Qs_MAPsum / period
    #sum_sl = np.sum([slmo_LES,slmo_COB,slmo_MAP])
    sum_sl = np.sum(slmo_MAP )
    #baselines_sum = np.sum([slbaseline_MAP, slbaseline_COB, slbaseline_LES])
    baselines_sum = np.sum(slbaseline_MAP)
    reduction_sum = baselines_sum - sum_sl
    perc_Qs = (reduction_sum)/baselines_sum
    return reduction_sum, perc_Qs

hockystick (outputdata,4, 13, 19, 2010-1985)
