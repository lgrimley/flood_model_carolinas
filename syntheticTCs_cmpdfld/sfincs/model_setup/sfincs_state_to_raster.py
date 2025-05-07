import os

import hydromt_sfincs.utils
import xarray as xr
import pandas as pd
from src.utils import calculate_flooded_area_by_process
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
import os
import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
mod.read()
results = mod.read_results()
os.chdir(base_root)
data = xr.open_dataset('sfincs_map.nc')

# ind = hydromt_sfincs.utils.read_binary_map_index('sfincs.ind')
# rst = hydromt_sfincs.utils.read_binary_map('sfincs.19990119.000000.rst',ind = ind, shape=grid.shape, mv=np.nan)
# da = xr.DataArray(rst, dims=("y", "x"), coords={"x": grid.x.values, "y": grid.y.values})
# da['spatial_ref'] = grid['spatial_ref']
