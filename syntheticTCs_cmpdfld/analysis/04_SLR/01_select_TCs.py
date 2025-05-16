import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS')


# Get the TC IDs for all the storms that generate a RP flood extent greater than 80 for all areas/scenarios
basins = ['Neuse', 'Pamlico','OnslowBay','CapeFear', 'LowerPeeDee', 'Domain']
storms = []
period = 'future'
for i in range(len(basins)):
    rp_file = fr'.\results_jointprob\basin_data\{basins[i]}_data_rp_{period}.csv'
    rp_data = pd.read_csv(rp_file, index_col=0)

    # Subset to storms of interest
    rp_threshold = 80
    for s in ['Runoff', 'Coastal','Compound', 'Total_Flooded']:
        fut_tcs = rp_data[rp_data[f'{s}_rp'] > rp_threshold]
        fut_tcs.to_csv(fr'.\SLR_analysis\{basins[i]}_data_rp_{period}_SLRsubset_{s}.csv')
        storms = storms + fut_tcs.index.tolist()

storms_f = np.unique(storms)
#pd.DataFrame(storms_f).to_csv(fr'.\SLR_analysis\SLR_storm_TCIDs.csv')
