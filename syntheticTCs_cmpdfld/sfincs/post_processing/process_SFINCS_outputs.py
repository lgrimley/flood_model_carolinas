import hydromt
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
from pathlib import Path
from src.core import TCFloodHazard
import xarray as xr

# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')

base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r')
mod.read()



















