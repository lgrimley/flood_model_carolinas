import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound\tables'
os.chdir(wdir)

past_df = pd.read_csv(r'ncep_AEP_floodStats_compound_sbgRes5m_hmin0.05m.csv', index_col=0)
past_df['area']


fut_df = pd.read_csv(r'canesm_ssp585_AEP_floodStats_compound_sbgRes5m_hmin0.05m.csv', index_col=0)

