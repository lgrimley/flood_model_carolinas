import matplotlib.pyplot as plt
import hydromt
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
import sys

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
from src.utils import *


# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')



# Load in the data catalogs needed for building the model
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_sfincs_Carolinas_Ch3 = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas_Ch3.yml')

# Setup working directory and model root, create and instance of a SFINCS model to write to
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup')
root = 'base_model'
mod = SfincsModel(root=root, mode='r+',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas, yml_sfincs_Carolinas_Ch3])
cat = mod.data_catalog

tc_id = 2773
# Setup directory information
stormData_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis'
track_file = os.path.join(stormData_root, r'.\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100')
wl_root = os.path.join(stormData_root, 'stormTide', 'adcirc_waterlevel_netcdf')
precip_root = os.path.join(stormData_root, 'rain', '03_TCR_RainOutput_Gridded_hourly')
wind_root = os.path.join(stormData_root, 'wind', '02_CLE15_WindOutput_Gridded')


''' Track Time information '''
tc_tracks = sio.loadmat(f'{track_file}.mat')
df = get_track_info_in_df(tc_id=tc_id, tc_tracks=tc_tracks)
time = df['datetime'][df['datetime'] != 0]

'''  Data '''
fname = f'{tc_id}.nc'
wl = cat.get_geodataset(os.path.join(wl_root, fname), crs=4326)
precip = xr.open_dataset(os.path.join(precip_root, fname))
wind = xr.open_dataset(os.path.join(wind_root, fname))
wind = wind.assign_coords(spatial_ref=precip['spatial_ref'])
wind = wind.resample(time='1H').ffill()

# Updating config
tstart = pd.to_datetime(wl['time'].min().values.astype(str)).strftime('%Y%m%d %H%M%S')
tstop = pd.to_datetime(wl['time'].max().values.astype(str)).strftime('%Y%m%d %H%M%S')
mod.setup_config(**{"tref": tstart, "tstart": tstart, "tstop": tstop})
mod.write_config(config_fn='sfincs.inp')
print(np.max(wl.sel(index=wl.index.values[0]).values))

if precip['time'].max() < wl['time'].max():
    print('Extending precip time series with zeros...')
    precip = precip.reindex(time=wl.time).fillna(0)
if wind['time'].max() < wl['time'].max():
    print('Extending wind time series with zeros...')
    wind = wind.reindex(time=wl.time).fillna(0)


# Write data as SFINCS input
SLR = 0.0
indices_to_remove = wl.where(abs(wl) > 100).dropna(dim='index').coords['index'].values
cleaned_wl = wl.drop_sel(index=indices_to_remove)
#cleaned_wl.assign_coords(spatial_ref = cleaned_wl.attrs['spatial_ref'])
wl = cleaned_wl + SLR
print(np.max(wl.sel(index=wl.index.values[0]).values))

mod.setup_waterlevel_forcing(geodataset=wl,
                             offset='lmsl_to_navd88',
                             timeseries=None, locations=None,
                             buffer=10000, merge=False
                             )

mod.setup_precip_forcing_from_grid(precip=precip, aggregate=False)
mod.setup_wind_forcing_from_grid(wind=wind)
mod.write_forcing()
_ = mod.plot_forcing(fn_out="forcing.png")
plt.close()






