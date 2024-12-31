import matplotlib.pyplot as plt
import xarray as xr
import scipy.io as sio
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.utils import *
from src.core import *


''' DATA to netcdf '''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585')
gcms = ['canesm_ssp585cal', 'cnrm6_ssp585cal', 'ecearth6_ssp585cal', 'ipsl6_ssp585cal', 'miroc6_ssp585cal']

for gcm in gcms:
    input_dir = os.path.join(os.getcwd(), 'wind', f'CLE15_Gridded_{gcm}')
    output_dir = os.path.join(os.getcwd(), 'wind', f'CLE15_ReGridded_{gcm}')

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    fname = rf'.\tracks\UScoast6_AL_{gcm}_roEst1rmEst1_trk100'
    tc_tracks = sio.loadmat(f'{fname}.mat')

    for file in os.listdir(input_dir):
        tc_id = int(file.strip('.nc'))
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

