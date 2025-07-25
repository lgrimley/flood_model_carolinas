import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\01_return_period_tables')


# Get the TC IDs for all the storms that generate a RP flood extent greater than 80 for all areas/scenarios
basins = ['Neuse', 'Pamlico','OnslowBay','CapeFear', 'LowerPeeDee', 'Domain']
storms = []
period = 'canesm'
for i in range(len(basins)):
    rp_file = fr'{basins[i]}_data_rp_{period}.csv'
    rp_data = pd.read_csv(rp_file, index_col=0)

    # Subset to storms of interest
    rp_threshold = 80
    for s in ['Runoff_Area_sqkm', 'Coastal_Area_sqkm','Compound_Area_sqkm', 'Total_Area_sqkm']:
        fut_tcs = rp_data[rp_data[f'{s}_RP'] > rp_threshold]
        #fut_tcs.to_csv(fr'..\05_SLR\subset_v2\{basins[i]}_data_rp_{period}_SLRsubset_{s}.csv')
        storms = storms + fut_tcs.index.tolist()

storms_f = np.unique(storms)
#pd.DataFrame(storms_f).to_csv(fr'..\05_SLR\subset_v2\SLR_storm_TCIDs.csv')

check = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\05_SLR\subset_v1\SLR_storm_TCIDs.csv', index_col=0)
list1 = storms_f.astype(int).tolist()
list2 = check['0'].astype(int).tolist()
common_values = [x for x in list1 if x in list2]
missing = [x for x in list1 if x not in list2]