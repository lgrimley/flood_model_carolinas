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
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml2 = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')

# Working directory and model root
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS')
root = 'ENC_200m_sbg5m_avgN_eff75_v2'
mod = SfincsModel(root=root, mode='r+', data_libs=[yml, yml2])
mod.read()
cat=mod.data_catalog


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
lulc = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\data\ICLUS_2100_585.tif')
mod.setup_cn_infiltration_with_ks(lulc=lulc,
                                  hsg='gNATSGO_hsg_conus',
                                  ksat='gNATSGO_ksat_DCP_0to20cm_carolinas',
                                  reclass_table=r'Z:\Data-Expansion\users\lelise\data\CN_Table_HSG_ICLUS_orig.csv',
                                  effective=0.75,
                                  block_size=2000)
mod.write()

