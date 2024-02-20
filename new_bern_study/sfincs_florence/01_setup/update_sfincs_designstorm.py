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
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
base_root = r'Z:\users\lelise\projects\HCFCD\sfincs_models\03_for_TAMU\Harvey\hcfcd_100m_sbg5m_v2_ksat25'

# Rainfall timeseries
rr_hr = pd.read_csv(r'Z:\users\lelise\projects\HCFCD\design_storms\hcfcd_future_designstorms.csv')
rr_hr['datetime'] = pd.to_datetime(rr_hr['datetime'])
rr_hr.set_index('datetime', drop=True, inplace=True)

# Loop through and create models
out_dir = r'Z:\users\lelise\projects\HCFCD\sfincs_models\03_for_TAMU\DesignStorms\mod_v3'
for rp in rr_hr.columns:
    base_root = os.path.join(out_dir, f'hcfcd_100m_sbg5m_v2_ksat25_F{rp}yr')
    mod = SfincsModel(root=base_root, mode='r+', data_libs=yml)
    # mod.update(os.path.join(out_dir, f'hcfcd_100m_sbg5m_v2_ksat25_F{rp}yr'), write=True)
    # 
    # # Updating config
    # mod.setup_config(
    #     **{
    #         "tref": "20220101 000000",
    #         "tstart": "20220101 000000",
    #         "tstop": "20220102 000000",
    # 
    #         'dtrstout': '259200',
    #         'dtout': '1800',
    #         'dthisout': '900',
    #         'tspinup': '86400',
    # 
    #         'advection': '1',
    #         'alpha': '0.5',
    #         'theta': '1',
    #         'huthresh': '0.05',
    #         'viscosity': '1',
    # 
    #         'min_lev_hmax': '-20',
    #         'zsini': '0',
    #         'stopdepth': '100',
    #         #'scsfile': '',
    #         #'netamprfile': ''
    #     }
    # )
    # print(mod.config)
    # mod.write_config(config_fn='sfincs.inp')
    # 
    # Setup water level forcing
    mod.setup_waterlevel_forcing(geodataset='hcfcd_designstorm_bc_waterlevel',
                                 timeseries=None, locations=None,
                                 buffer=10000, merge=False)
    mod.write_forcing(data_vars='bzs')
    bzs = mod.forcing['bzs']
    print('Write bzs')

    # Setup discharge forcing
    mod.setup_discharge_forcing(geodataset='hcfcd_designstorm_bc_discharge', merge=False)
    mod.write_forcing(data_vars='dis')
    print('Write dis')

    # Write uniform precip
    df_ts = rr_hr[rp].squeeze()
    df_ts.name = "precip"
    df_ts.index.name = "time"
    mod.set_forcing(df_ts.to_xarray(), name="precip")

    _ = mod.plot_forcing(fn_out="forcing.png")
    plt.close()
    print('Plot forcing')

    mod.write()

