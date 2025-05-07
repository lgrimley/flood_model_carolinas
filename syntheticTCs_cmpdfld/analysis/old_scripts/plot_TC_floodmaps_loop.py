import os
import xarray as xr
from hydromt_sfincs import SfincsModel
import scipy.io as sio
from src.utils import track_points_to_linestring, plot_storm_maps


# Load in the data catalogs needed for building the model
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
sfincs_mod = SfincsModel(root=r'.\03_MODEL\sfincs_base_mod', mode='r', data_libs=yml_base)

# Load the TC Track file
fname = r'.\02_DATA\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')
tc_tracks_polyline = track_points_to_linestring(tc_tracks, output_shpfile=None)

# Change the directory to the model results
results_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\ncep_runs\tests'
runs_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\ncep_runs\tests'
zsmax_ds = xr.open_dataarray(os.path.join(results_dir, 'zsmax_tests.nc'))
zsmax_ds = zsmax_ds.to_dataset(dim='scenario')
attr_ds_all = xr.open_dataset(os.path.join(results_dir, 'attribution_tests.nc'))
output_directory = os.path.join(results_dir, 'figs')
if os.path.exists(output_directory) is False:
    os.makedirs(output_directory)

for tc_id in zsmax_ds.tc_id.values:
    tc_track_gdf = tc_tracks_polyline[tc_tracks_polyline['tc_id'] == tc_id]
    tc_track_gdf = tc_track_gdf.to_crs(32617)
    boundary_condition_dir = os.path.join(runs_dir, f'TC_{str(tc_id).zfill(4)}', 'sfincs_bc_inputs')
    attr_ds = attr_ds_all.sel(tc_id=tc_id)
    zsmax_da =zsmax_ds.sel(tc_id=tc_id)['compound']

    _  = plot_storm_maps(tc_id, sfincs_mod, tc_track_gdf, zsmax_da, attr_ds, boundary_condition_dir, output_directory)
    print(tc_id)


