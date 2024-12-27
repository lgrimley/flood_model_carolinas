import matplotlib.pyplot as plt
import xarray as xr
import scipy.io as sio
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.utils import *
from src.core import *


fname = r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')

''' DATA to netcdf '''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585')
gcms = ['canesm_ssp585cal', 'cnrm6_ssp585cal', 'ecearth6_ssp585cal', 'ipsl6_ssp585cal', 'miroc6_ssp585cal']
missing = []

input_dir = os.path.join(os.getcwd(), 'CLE15_Gridded')
output_dir = os.path.join(os.getcwd(), 'CLE15_WindOutput_Gridded')
# figs_dir = os.path.join(os.getcwd(), 'Figs')
# usa = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k'
#                     r'\cb_2018_us_state_500k.shp')

select_tcs = pd.read_csv(r'..\tracks\adcirc_modeled_TCs_all.csv')
for tc_id in select_tcs['tc_id'].values:
    file = f'{str(tc_id).zfill(4)}.nc'
    out_name = os.path.join(output_dir, file)

    if os.path.exists(out_name) is False:
        print(tc_id)

        # Get the track information w/ datetime
        df = get_track_info_in_df(tc_id=tc_id, tc_tracks=tc_tracks)
        time = df['datetime'][df['datetime'] != 0]

        # Load the gridded TCR rainfall for the TC
        data = xr.open_dataset(os.path.join(input_dir, file))
        data['time'] = time.values
        data = data.rio.write_crs("epsg:4326", inplace=True)

        # Resample to hourly data (this might not be necessary for hydromt)
        # data = data.resample(time='1H').ffill()
        data.to_netcdf(out_name)

        # Plotting
        # track_gdf = track_as_gdf(df)
        # dplot = data.max(dim='time')['wind_speed']
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)
        # dplot.plot(ax=ax, cmap='jet')
        # track_gdf.geometry.plot(ax=ax, color='black')
        # usa.geometry.plot(ax=ax, color='none', edgecolor='black')
        # ax.set_title(tc_id)
        # plt.savefig(os.path.join(figs_dir, file.replace('.nc', '_max_windSpd.png')))
        # plt.close()

