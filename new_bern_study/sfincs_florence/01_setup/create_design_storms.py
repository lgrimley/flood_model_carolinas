import os
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime as dt
import rioxarray as rio
import xarray as xr
import numpy as np
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils
import datetime

''' Get future IDF curve scaling info '''

os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\model_data\design_storms_setup')
# Load CSV downloaded from NOAA PFQ website
pf_atlas14_mm = pd.read_csv('PF_Depth_Metric_PDS.csv',
                            skiprows=13,
                            skipfooter=2)
pf_atlas14_mm.columns = pf_atlas14_mm.columns.str.replace(' ', '')
# Duration of the design storms in minutes
pf_atlas14_mm['durations_min'] = [5, 10, 15, 30, 60,
                                  120, 180, 360, 720, 1440,
                                  2880, 4320, 5760, 10080, 14400, 28800, 43200, 64800, 86400]
pf_atlas14_mm.set_index('durations_min', drop=True, inplace=True)
pf_atlas14_mm.drop(columns=['bydurationforARI(years):'], inplace=True)

# Create SCS Type3 hyetographs
# Load the SCS Type 3 curve
scsType3 = pd.read_csv('scs_type3.csv')
scsType3['time (hr)'] = pd.to_timedelta(scsType3['time (hr)'], unit='hour')
scsType3.set_index('time (hr)', inplace=True, drop=True)
tstart = datetime.datetime(2022, 1, 1, 0, 0)

# NOAA Atlas 14 Design Storm Time Series
duration = 24 * 60
rain_rate = pd.DataFrame()
rain_rate['time_hr'] = scsType3.index
rain_rate.set_index('time_hr', inplace=True, drop=True)
for ari in pf_atlas14_mm.columns:
    tot_prcp = pf_atlas14_mm[pf_atlas14_mm.index == duration][ari].item()
    scsType3[ari] = scsType3['fraction of 24-hr rainfall'] * tot_prcp
    rain_rate[ari] = scsType3[ari].diff(periods=1)

rain_rate.iloc[0, :] = rain_rate.iloc[1, :]
rain_rate['datetime'] = rain_rate.index + tstart
rain_rate.set_index('datetime', inplace=True, drop=True)
rr_hr = rain_rate.resample(rule='h').sum()
rr_hr.to_csv('present_24hr_designstorms_noaa_atlas14.csv', index=True)

''' Future IDF curves '''
future_idf_filepath = r'Z:\users\lelise\projects\HCFCD\design_storms\future_designstorms\data'
proj_precip = xr.open_dataset(os.path.join(future_idf_filepath, '03_Projected-Precipitation.nc'))
# Explore the data
print(proj_precip['period'].data)
print(proj_precip['ari'].data)
# extract future precip data from center of New Bern
lat, lon = [35.110034, -77.056639]
da = proj_precip.sel(lon=lon, lat=lat, method='nearest')['pr_projected']
df = da.to_pandas()
df['500'] = 0  # Future IDF doesn't have 500-yr so set to 0

# Load future relative change netcdf and plot
rc = xr.open_dataset(os.path.join(future_idf_filepath, '01_Relative-change.nc'))
da = rc.sel(lon=lon, lat=lat, method='nearest')['relative_change']
df = da.to_pandas()
fig, ax = plt.subplots(tight_layout=True, figsize=(5.5, 3))
df.plot(ax=ax, linewidth=1.5, linestyle='-.')
ax.set_ylabel('Relative Change (%)\n in 24-hr Total Precipitation')
ax.set_xlabel('Future Period')
ax.legend(title='ARI', loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('IDF_24hrStorm_future_relative_change_NBLL.png')
plt.close()
# Output scaled storm totals for future IDF curves
round(df, 4).to_csv('nbll_aic_idf_relative_change.csv')


# Scale NOAA Atlas 14 to get total precip for future design storms
future_cum_precip = df.copy()
for col in future_cum_precip.columns:
    print(col)
    future_cum_precip[col] = future_cum_precip[col] * pf_atlas14_mm[str(col)][pf_atlas14_mm.index == duration].item()

''' Write Design Storms Boundary Condition Files for SFINCS '''
# Filepath to data catalog yml
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
base_root = r'Z:\users\lelise\projects\NBLL\sfincs\design_storms\nbll_40m_sbg3m_v3_eff25_designstorm'

# Loop through and create models
out_dir = r'Z:\users\lelise\projects\NBLL\sfincs\design_storms'
for rp in rr_hr.columns:
    mod = SfincsModel(root=base_root, mode='r', data_libs=yml)
    out_mod = os.path.join(out_dir, f'nbll_40m_sbg3m_v3_eff25_P{rp}yr')
    mod.update(out_mod, write=True)

    # Updating config
    # mod.setup_config(
    #     **{
    #         "tref": "20220101 000000",
    #         "tstart": "20220101 000000",
    #         "tstop": "20220102 000000",
    #         'dtout': '1800',
    #         'tspinup': '86400',
    #     }
    # )
    # print(mod.config)
    # mod.write_config(config_fn='sfincs.inp')
    #
    # # Setup water level forcing
    # mod.setup_waterlevel_forcing('nbll_designstorm_bc_waterlevel',
    #                              timeseries=None,
    #                              locations=None,
    #                              buffer=10000,
    #                              merge=False)
    # mod.write_forcing(data_vars='bzs')
    # print('Write bzs')

    # Write uniform precip
    df_ts = rr_hr[rp].squeeze()
    df_ts.name = "precip"
    df_ts.index.name = "time"
    mod.set_forcing(df_ts.to_xarray(), name="precip")

    _ = mod.plot_forcing(fn_out="forcing.png")
    plt.close()
    print('Plot forcing')

    mod.write()

