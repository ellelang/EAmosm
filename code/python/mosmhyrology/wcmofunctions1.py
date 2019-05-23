from pathlib import Path
import numpy as np
import pandas as pd

mdata_folder = Path("C:/Users/langzx/Documents")

wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
WCMO_N = wcmo.shape[0]
wcmo['select'] = np.random.randint(2, size=WCMO_N)
flowdata = pd.read_csv(mdata_folder/"SWAT_WY.csv")
SBIO = pd.read_csv(mdata_folder/"SB IO Detail.csv")
subbasin = flowdata['Subbasin']

# Section 1: Allocate WCMO to each HYDSB
def allocate (mo, sb_n, mo_n):
  SB_mocount = np.zeros(sb_n)
  SB_moaream2 = np.zeros(sb_n)
  mo['select'] = np.random.randint(2, size=mo_n)
  for i in range (0,sb_n):
      for j in range (0,mo_n):
          if mo['HYDSB_LES30SB'][j] == i and mo['select'][j] == 1:
             SB_mocount [i] =  SB_mocount [i] + 1
             SB_moaream2 [i] = SB_moaream2 [i] + mo['area_m2'][j]
  return SB_mocount,SB_moaream2

WCMOselect =  allocate (wcmo,30,WCMO_N)

# Section 2: Call water yield data
WaterYeild = flowdata['Water Yield']

# Section 3: River routing (MG-CG)

SBarea_m2 = SBIO['SBarea_m2']
SB_mocount = allocate (wcmo,30,WCMO_N) [0]
SB_moaream2 = allocate (wcmo,30,WCMO_N) [1]

def riverrout (WaterYeild, SBarea_m2,SB_mocount,SB_moaream2):
    C = 2
    Anmin_factor = 5/100
    alpha = 0.5
    k = 1e-7
    ET = 1.16e-8
    Anmin = SBarea_m2 * Anmin_factor
    WCMO_D = 6.6 * 0.3048
    WCMO_DAfactor = 8.9
    dt = 86400
    n = 30 * 9131
    Aw = np.empty(30)
    Awbar = np.empty (30)
    WCMO_SAbar = np.empty (30)
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

    for i in range (0,30):
        if SB_mocount [i] == 0:

          for j in range (0, 9131):
                index = j + ((i + 1) - 1) * 9131
                Qtnout[index] = WaterYeild[index]
                #Qnout[index] = index
### If WCMO is selected:   

#draiange area as an exponential function to show diminishing return
#Aw(i) = (SBarea(i) - Anmin) * (1 - Exp(-1e-07 * WCMOsumarea(i))) 'calculate drainage area of selected WCMO
#drainage area as a linear function of SA#             
        else:

            if WCMO_DAfactor * SB_moaream2 [i] > SBarea_m2[i] - Anmin [i]:
                  Aw [i] = SBarea_m2[i] - Anmin [i]
            else:
                  Aw [i] = WCMO_DAfactor * SB_moaream2 [i]

            Awbar [i] = Aw [i] / SB_mocount [i] # average drainage area for selected WCMOs

            WCMO_SAbar [i] =  SB_moaream2 [i] / SB_mocount [i] # average surface area of WCMOSs selected in a SB in m2

            circ = np.sqrt(WCMO_SAbar [i]/np.pi) * 2 * np.pi # '[m] circuference of the average wetland

            L = alpha * circ # [m] effective length of crest

              ##Set initial condition of water
            index_initial = 0 + ((i + 1) - 1) * 9131

            It [index_initial] = WaterYeild[index_initial] * (Awbar [i] / SBarea_m2[i]) # m3/s] inflow to wetland is proportional to flow in SB--> Q/I = SBarea/(wetland drainge area)-->I=Q*(wetland drainage area)/SBarea

            St [index_initial] = 0 # initial condition empty WCMO

            Ht [index_initial] = St [index_initial] / WCMO_SAbar [i] # [m] = 0m

            Qt [index_initial] = 0 # C * designLC * Ht(1, i) ^ 1.5 '[cms]

              ### iteration begins:
            for j in range (1,9131): # for all flow record from 1985-2009,j=0-->1/1/1985

                  index_2 = j + ((i + 1) - 1) * 9131

                  It [index_2] = WaterYeild[index_2] * (Awbar [i] / SBarea_m2[i]) #[m3/s][-][m2]/[m2]=[m3/s]
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
    ### When there are WCMO selected
Qtnout_record = riverrout(WaterYeild, SBarea_m2,SB_mocount,SB_moaream2)
Qtnout_record 

# Section 3.2: Apply River_routing
 
sigma = 0.7
n = 30 * 9131
Headerwater = SBIO['Headwater']
upSBs = SBIO['up_SBs']
ID_next = SBIO['ID_next']
C1 = SBIO['C1']
C2 = SBIO['C2']
C3 = SBIO['C3']
def apply_riverrout (sigma,n,Qtnout,Headerwater,upSBs, ID_next,C1,C2,C3):
     count = np.zeros (30)
     Qtnoutriver = np.zeros (n)
     QtnoutriverMCrouted = np.zeros (n)
     #######HEADWATER: move down river starting from upppermost SB
     for m in range(29, -1, -1):
         if Headerwater[m] == "Y":
             i = m
             count [i] = count [i] + 1
             # call storageoutflow function to generate Qtnout (index)
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
             ##########END HEADWATER
             ###MOVE DOWN SB UNTIL REACHING THE TERMINAL
             o = i
             i = ID_next[o]
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
outputdata.to_csv(mdata_folder/"output.csv", index=False)

## Section 4
def hockystick (outputdata,les_sb, cob_sb, map_sb,period):
    Qth = 0.01
    # Drainage area
    LES_DA = 1156.56
    COB_DA = 784.12
    MAP_DA = 877.15
    #Incised length
    LES_L = 40.84
    COB_L = 35.79
    MAP_L = 31.91

    CQ_a = 6867.9
    CQ_b = 2.1572

    Qs_LES  = np.zeros(9131)
    Qs_COB  = np.zeros (9131)
    Qs_MAP  = np.zeros (9131)

    Qs_LESsum = 0
    Qs_COBsum = 0
    Qs_MAPsum = 0

    slbaseline_MAP = 15516
    slbaseline_COB = 17557
    slbaseline_LES = 18771

    SBoutput_LES = outputdata.loc[outputdata['Subbasin'] == les_sb].reset_index(inplace=False)
    SBoutput_COB = outputdata.loc[outputdata['Subbasin'] == cob_sb].reset_index(inplace=False)
    SBoutput_MAP = outputdata.loc[outputdata['Subbasin'] == map_sb].reset_index(inplace=False)


    for j in range (0,9131):
        # populate LES/COB/MAP sediment loading from incised zone
        if SBoutput_LES['QtnoutriverMCrouted'][j] > Qth * LES_DA:
            Qs_LES [j] = CQ_a * (SBoutput_LES['QtnoutriverMCrouted'][j] / LES_DA)**CQ_b * LES_L
        else:
            Qs_LES [j] = 0

        if SBoutput_COB['QtnoutriverMCrouted'][j] > Qth * COB_DA:
            Qs_COB [j] = CQ_a * (SBoutput_COB['QtnoutriverMCrouted'][j] / LES_DA)**CQ_b * COB_L
        else:
            Qs_COB [j] = 0

        if SBoutput_MAP['QtnoutriverMCrouted'][j] > Qth * MAP_DA:
            Qs_MAP [j] = CQ_a * (SBoutput_MAP['QtnoutriverMCrouted'][j] / MAP_DA)**CQ_b * MAP_L
        else:
            Qs_MAP [j] = 0

        Qs_LESsum = Qs_LESsum + Qs_LES [j]
        Qs_COBsum = Qs_COBsum + Qs_COB [j]
        Qs_MAPsum = Qs_MAPsum + Qs_MAP [j]

    slmo_LES  = Qs_LESsum / period
    slmo_COB  = Qs_COBsum / period
    slmo_MAP  = Qs_MAPsum / period
    sum_sl = np.sum([slmo_LES,slmo_COB,slmo_MAP])
    baselines_sum = np.sum([slbaseline_MAP, slbaseline_COB, slbaseline_LES])
    reduction_sum = baselines_sum - sum_sl
    perc_Qs = (reduction_sum)/baselines_sum
    return reduction_sum, perc_Qs

hockystick (outputdata,4, 13, 19, 2010-1985)