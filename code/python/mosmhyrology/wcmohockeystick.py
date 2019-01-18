
from pathlib import Path
import numpy as np
import pandas as pd
mdata_folder = Path("C:/Users/langzx/Documents")

wcmo = pd.read_csv(mdata_folder/"WCMO.csv")
flowdata = pd.read_csv(mdata_folder/"SWAT_WY.csv")
SBIO = pd.read_csv(mdata_folder/"SB IO Detail.csv")
SBoutput = pd.read_csv(mdata_folder/"Output Subbasin Daily.csv")

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

SBoutput_LES = SBoutput.loc[SBoutput['Subbasin'] == 4].reset_index(inplace=False)
SBoutput_COB = SBoutput.loc[SBoutput['Subbasin'] == 13].reset_index(inplace=False)
SBoutput_MAP = SBoutput.loc[SBoutput['Subbasin'] == 19].reset_index(inplace=False)


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

period = 2010-1985
slmo_LES  = Qs_LESsum / period
slmo_LES
slmo_COB  = Qs_COBsum / period
slmo_COB
slmo_MAP  = Qs_MAPsum / period
slmo_MAP

slbaseline_MAP = 15516
slbaseline_COB = 17557
slbaseline_LES = 18771

baseline_sum = np.sum([slbaseline_MAP, slbaseline_COB, slbaseline_LES])
baseline_sum
reduction_sum = np.sum([slmo_LES, slmo_LES, slmo_MAP])
reduction_sum

def Reduction_perc (baselines_sum, current_sum):
    perc_Qs = (baseline_sum - current_sum)/baseline_sum
    return perc_Qs
