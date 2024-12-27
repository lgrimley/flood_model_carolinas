from hydromt_sfincs import SfincsModel
import sys
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld\src')
from src.core import *
import shutil
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

arg1 = sys.argv[1]
tc_index = int(arg1)
track_dir = sys.argv[2]

# Setup working directory and model root, create and instance of a SFINCS model to write to
# Load in the data catalogs needed for building the model
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_sfincs_Carolinas_Ch3 = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas_Ch3.yml')
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL')
# Load the SFINCS base model
mod = SfincsModel(root=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_base_mod', mode='r',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas, yml_sfincs_Carolinas_Ch3])
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

tc_index = 5008
# Load the data catalog where track data is saved
DatCat = hydromt.DataCatalog(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\data_catalog_NCEP.yml')

try:
    track = SyntheticTrack(DataPaths=NCEP_DataPaths, DatCat=DatCat, tc_index=tc_index)

    # Updating config file for storm times
    mod.setup_config(**{"tref": track.time_range[0], "tstart": track.time_range[0], "tstop": track.time_range[1]})
    mod.write_config(config_fn='sfincs.inp')
    mod.setup_waterlevel_forcing(geodataset=track.merged_waterlevel, offset='lmsl_to_navd88', buffer=100000, merge=False)

    # Get the gages where no stormtide data is available and pull the tides
    tide_stations = track.tide_offset_table[track.tide_offset_table['st_ht_dt'].isna()]['Point'].tolist()
    print(f'Tides used where ADCIRC stormtide output is not available at gages: {tide_stations}')
    if len(tide_stations) > 0:
        tide_stations_int = [int(s.split('P')[1]) for s in tide_stations]
        tide_filler_data = track.reanalysis_data_offset.sel(index=tide_stations)
        tide_filler_data = tide_filler_data.assign_coords(index=tide_stations_int)
        mod.setup_waterlevel_forcing(geodataset=tide_filler_data, offset='lmsl_to_navd88', buffer=100000, merge=True)
    x = len(mod.forcing['bzs'].index.values)
    print(f'Total number of gage inputs for the track: {x}')

    # Load and write the meteo output
    mod.setup_precip_forcing_from_grid(precip=track.precip, aggregate=False)
    mod.setup_wind_forcing_from_grid(wind=track.wind)
    print(f'Wind and rainfall SFINCS inputs created.')

    print('Writing the boundary condition files.')
    mod.write_forcing()
    # _ = mod.plot_forcing(fn_out=r"..\forcing.png")
    # plt.close()

    # Now we move the files to the track directory
    files_2_move = ['precip_2d.nc', 'wind_2d.nc', 'sfincs.bnd', 'sfincs.bzs', 'sfincs.inp']#,'forcing.png']
    for file in files_2_move:
        file_src = os.path.join(mod.root, file)
        file_dst = os.path.join(track_dir, file)
        _ = shutil.copyfile(file_src, file_dst)
        print(f'Created {file_dst}')

    # Now generate the tide only inputs and plot the forcing (used in runoff only scenario)
    tides_only = track.reanalysis_data.sel(index=track.coastal_locations_mapped['Point'].tolist())
    tide_stations_int = [int(s.split('P')[1]) for s in track.coastal_locations_mapped['Point'].tolist()]
    tides_only2 = tides_only.assign_coords(index=tide_stations_int)
    mod.setup_waterlevel_forcing(geodataset=tides_only2, offset='lmsl_to_navd88', buffer=100000, merge=False)
    mod.write_forcing()
    print('Tides only bnd and bzs files created.')
    # _ = mod.plot_forcing(fn_out=r"..\forcing.png")
    # plt.close()
    files_2_move = ['sfincs.bnd', 'sfincs.bzs']#, 'forcing.png']
    for file in files_2_move:
        file_src = os.path.join(mod.root, file)
        file_dst = os.path.join(track_dir, f'tides_{file}')
        _ = shutil.copyfile(file_src, file_dst)
        print(f'Created {file_dst}')

    # # Plot the track and total precipitation
    # print('Plotting the total precipitation and storm track.')
    # track_geom = track.track_points_to_linestring
    # track_geom = track_geom.to_crs(32617)
    # fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(6, 8), subplot_kw={'projection': utm})
    # ax.add_feature(cfeature.COASTLINE, zorder=2)
    # cm = mod.forcing['precip_2d'].sum(dim='time').plot(ax=ax, cmap='jet', add_colorbar=False)
    # mod.region.plot(ax=ax, color='none', edgecolor='black')
    # track_geom.plot(ax=ax, color='black', linewidth=3)
    # pos0 = ax.get_position()
    # cax = fig.add_axes([pos0.x1 + 0.3, pos0.y0+0.15, 0.05, pos0.height * 0.6])
    # cbar = fig.colorbar(cm,
    #                     cax=cax,
    #                     orientation='vertical',
    #                     extend='max',
    #                     label = 'Total Precip (mm)'
    #                     )
    # ax.set_title(str(tc_index))
    # minx, miny, maxx, maxy = mod.region.total_bounds
    # ax.set_xlim(minx, maxx)
    # ax.set_ylim(miny, maxy)
    # plt.savefig(os.path.join(track_dir, f'track_totalPrecip.png'), dpi=300, bbox_inches="tight")
    # plt.close()
except:
    print(f'Issue with TC: {tc_index}')


