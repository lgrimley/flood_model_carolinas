import os
import datetime
import hydromt
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Filepath to data catalog yml
#cat_dir = '/projects/sfincs/data'
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2018_Florence\mod_v6')
#os.chdir('/projects/sfincs/')
root = 'flo_hindcast_v6_100m_LPD2m_avgN_ksat_eff0.75'
mod = SfincsModel(root=root, mode='r+', data_libs=yml)

ks = cat.get_rasterdataset(os.path.join(mod.root, 'gis', 'ks.tif'))
mod.grid['ks'] = ks
mod.write_grid()


mod.update('flo_hindcast_v6_200m_LPD2m_avgN_ksat_eff0.75')

"""Setup model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS
including recovery term based on the soil saturation

Parameters
---------
lulc : str, Path, or RasterDataset
    Landuse/landcover data set
hsg : str, Path, or RasterDataset
    HSG (Hydrological Similarity Group) in integers
ksat : str, Path, or RasterDataset
    Ksat (saturated hydraulic conductivity) [mm/hr]
reclass_table : str, Path, or RasterDataset
    reclass table to relate landcover with soiltype
effective : float
    estimate of percentage effective soil, e.g. 0.50 for 50%
block_size : float
    maximum block size - use larger values will get more data in memory but can be faster, default=2000
"""

mod.setup_cn_infiltration_with_ks(lulc='nlcd_2016',
                                  hsg='surrgo_hsg_conus',
                                  ksat='surrgo_ksat_DCP_0to20cm_mm_carolinas',
                                  reclass_table=r'/projects/sfincs/data/soil/surrgo/CN_Table_HSG_NLCD.csv',
                                  effective=0.75,
                                  block_size=2000)
mod.write()

