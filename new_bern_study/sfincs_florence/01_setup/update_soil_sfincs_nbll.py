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
cat_dir = '/projects/sfincs/data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
os.chdir('/projects/sfincs')
root = 'nbll_40m_sbg3m_v3'
mod = SfincsModel(root=root, mode='r', data_libs=yml)
mod.update('nbll_40m_sbg3m_v3_eff25')

lulc = mod.data_catalog.get_rasterdataset('nlcd_2016', geom=mod.region)
hsg = mod.data_catalog.get_rasterdataset('surrgo_hsg_conus', geom=mod.region)
ksat = mod.data_catalog.get_rasterdataset('surrgo_ksat_DCP_0to20cm_mm_carolinas', geom=mod.region)

mod.setup_cn_infiltration_with_ks(lulc=lulc,
                                  hsg=hsg,
                                  ksat=ksat,
                                  reclass_table=os.path.join(cat_dir, 'soil/surrgo/CN_Table_HSG_NLCD.csv'),
                                  effective=0.25,
                                  block_size=2000)
print('Done with setting up CN infiltration with recovery')

# Setup Curve Number Infiltration
mod.setup_cn_infiltration(cn='gcn250',
                          antecedent_moisture='avg')
_ = mod.plot_basemap(fn_out='scs_curvenumber.png', variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
plt.close()
mod.write()
print('Done with setting up CN infiltration without recovery')

mod.write_grid()
mod.write()

