import os.path
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
import matplotlib.pyplot as plt

from src.core import *
import shutil
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')

# Setup working directory and model root, create and instance of a SFINCS model to write to
# Load in the data catalogs needed for building the model
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_sfincs_Carolinas_Ch3 = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas_Ch3.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup\base_model'
mod = SfincsModel(root=root, mode='r',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas, yml_sfincs_Carolinas_Ch3])
da = mod.grid['dep']
wkt = da.raster.crs.to_wkt()
utm_zone = da.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
DatCat = hydromt.DataCatalog(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\data_catalog_NCEP.yml')
for tc_index in [1234, 2645, 2773, 3429]:
    track = SyntheticTrack(DataPaths=NCEP_DataPaths, DatCat=DatCat, tc_index=tc_index)
    wind = track.wind
    precip = track.precip
    stormTide = track.stormTide
    tide_stations = track.tide_offset_table[track.tide_offset_table['st_ht_dt'].isna()]['Point'].tolist()
    print(tide_stations)

    # Write data as SFINCS input
    # Updating config
    mod.setup_config(**{"tref": track.time_range[0], "tstart": track.time_range[0], "tstop": track.time_range[1]})
    mod.write_config(config_fn='sfincs.inp')

    indices_to_remove = stormTide.where(abs(stormTide) > 100).dropna(dim='index').coords['index'].values
    waterlevel = stormTide.drop_sel(index=indices_to_remove)
    mod.setup_waterlevel_forcing(geodataset=waterlevel, offset='lmsl_to_navd88', buffer=2000, merge=False)
    if len(tide_stations) > 0:
        tide_stations_int = [int(s.split('P')[1]) for s in tide_stations]
        tide_filler_data = track.reanalysis_data.sel(index=tide_stations)
        tide_filler_data = tide_filler_data.assign_coords(index=tide_stations_int)
        mod.setup_waterlevel_forcing(geodataset=tide_filler_data, offset='lmsl_to_navd88', buffer=5000, merge=True)

    mod.setup_precip_forcing_from_grid(precip=precip, aggregate=False)
    mod.setup_wind_forcing_from_grid(wind=wind)
    mod.write_forcing()
    _ = mod.plot_forcing(fn_out=r"..\forcing.png")
    plt.close()

    track_dir = os.path.join(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup',f'TC_{str(tc_index).zfill(4)}')
    if os.path.exists(track_dir) is False:
        os.makedirs(track_dir)

    files_2_move = ['precip_2d.nc', 'wind_2d.nc', 'sfincs.bnd', 'sfincs.bzs', 'sfincs.inp', r'forcing.png']
    for file in files_2_move:
        file_src = os.path.join(mod.root, file)
        file_dst = os.path.join(track_dir, file)
        _ = shutil.copyfile(file_src, file_dst)
        print(f'Created {file_dst}')

    # Now generate the tide only inputs and plot the forcing (used in runoff only scenario)
    map = track.coastal_locations_mapped
    tides_only = track.reanalysis_data.sel(index=map['Point'].tolist())
    tide_stations_int = [int(s.split('P')[1]) for s in map['Point'].tolist()]
    tides_only2 = tides_only.assign_coords(index=tide_stations_int)

    mod.setup_waterlevel_forcing(geodataset=tides_only2, offset='lmsl_to_navd88', buffer=2000, merge=False)
    mod.write_forcing()
    _ = mod.plot_forcing(fn_out=r"..\forcing.png")
    plt.close()
    files_2_move = ['sfincs.bnd', 'sfincs.bzs', r'forcing.png']
    for file in files_2_move:
        file_src = os.path.join(mod.root, file)
        file_dst = os.path.join(track_dir, f'tides_{file}')
        _ = shutil.copyfile(file_src, file_dst)
        print(f'Created {file_dst}')

    # Plot the track and total precipitation
    track_geom = track.track_points_to_linestring
    track_geom = track_geom.to_crs(32617)
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True,
                           figsize=(6, 8), subplot_kw={'projection': utm})
    ax.add_feature(cfeature.COASTLINE, zorder=2)
    cm = mod.forcing['precip_2d'].sum(dim='time').plot(ax=ax, cmap='jet', add_colorbar=False)
    mod.region.plot(ax=ax, color='none', edgecolor='black')
    track_geom.plot(ax=ax, color='black', linewidth=3)
    pos0 = ax.get_position()
    cax = fig.add_axes([pos0.x1 + 0.3, pos0.y0+0.15, 0.05, pos0.height * 0.6])
    cbar = fig.colorbar(cm,
                        cax=cax,
                        orientation='vertical',
                        extend='max',
                        label = 'Total Precip (mm)'
                        )
    ax.set_title(tc_index)
    minx, miny, maxx, maxy = mod.region.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    plt.savefig(os.path.join(track_dir, f'track.png'), dpi=300, bbox_inches="tight")
    plt.close()

