import scipy.io as sio
from src.utils import track_points_to_linestring
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
import xarray as xr
import os
import numpy as np


res = 200
hmin = 0.05
proj_crs = 32617
target_tc_id=4197

# Load in the data catalogs needed for building the model
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
sfincs_mod = SfincsModel(root=r'.\03_MODEL_RUNS\sfincs_base_mod', mode='r', data_libs=data_catalog_yml)
# Read in the data catalog to get the model and basin geom
cat = sfincs_mod.data_catalog

# Load the TC Track file
fname = r'.\02_DATA\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')
tc_tracks_polyline = track_points_to_linestring(tc_tracks, output_shpfile=None)

wdir = r'.\04_MODEL_OUTPUTS'

# Load the basins geodataframe
basins = cat.get_geodataframe(data_like=rf'{wdir}/masks/basins_shp/huc6_basins.shp')
clip_geom = basins.to_crs(epsg=proj_crs)

# Chunk size specified for dask
chunks_size = {'x': 5000, 'y': 5000}

# Load the elevation dataarray
elevation_da = cat.get_rasterdataset(rf'.\03_MODEL_RUNS\subgrid\dep_subgrid_{res}m.tif', geom=clip_geom, chunks=chunks_size)
elevation_da.name = 'gnd_elevation'

# Load the water body mask dataarray
wb_mask = cat.get_rasterdataset(rf'{wdir}/masks/water_mask_sbgRes{res}m.tif', chunks=chunks_size, geom=clip_geom)
wb_mask.rio.write_crs(proj_crs, inplace=True)
wb_mask.name = 'wb_mask'

# Load the basin mask dataarray
basin_mask = cat.get_rasterdataset(rf'{wdir}/masks/basin_mask_sbgRes{res}m.tif', crs=proj_crs, chunks=chunks_size, geom=clip_geom)
basin_mask.rio.write_crs(proj_crs, inplace=True)
basin_mask.name = 'basin_mask'

# Your NetCDF file paths
# Load the water levels
zsmax_filepath=os.path.join(os.getcwd(), wdir, 'ncep', 'zsmax')
files = os.listdir(zsmax_filepath)
for file in files:
    print(f"Checking {file}...")
    # Open the file without loading data into memory
    ds = xr.open_dataset(os.path.join(zsmax_filepath, file))
    # Assume tc_id is a coordinate or variable
    # Adjust this if tc_id is a coordinate: ds.coords['tc_id']
    tc_ids = ds['tc_id'].values

    if target_tc_id in tc_ids:
        print(f"Found {target_tc_id} in {file}")
        # Use boolean indexing to extract only the relevant slice
        zsmax_ds = cat.get_rasterdataset(os.path.join(zsmax_filepath, file), crs=proj_crs,
                                         geom=clip_geom, chunks=chunks_size)
        break  # Stop after finding the match
    ds.close()  # Clean up if not using a context manager
else:
    print(f"{target_tc_id} not found in any file.")

# Load the attribution
attr_filepath= os.path.join(os.getcwd(), wdir, 'ncep', 'attribution')
files = os.listdir(attr_filepath)
for file in files:
    print(f"Checking {file}...")
    # Open the file without loading data into memory
    ds = xr.open_dataset(os.path.join(attr_filepath, file))
    # Assume tc_id is a coordinate or variable
    # Adjust this if tc_id is a coordinate: ds.coords['tc_id']
    tc_ids = ds['tc_id'].values

    if target_tc_id in tc_ids:
        print(f"Found {target_tc_id} in {file}")
        # Use boolean indexing to extract only the relevant slice
        attr_ds = cat.get_rasterdataset(os.path.join(attr_filepath, file), crs=proj_crs, geom=clip_geom, chunks=chunks_size)
        break  # Stop after finding the match
    ds.close()  # Clean up if not using a context manager
else:
    print(f"{target_tc_id} not found in any file.")

# Selected run, mask out data beyond the shapefile
zsmax_da = zsmax_ds.sel(tc_id=target_tc_id, scenario='compound')
attr_da = attr_ds.sel(tc_id=target_tc_id)['zsmax_attr']

# Clean up the data array before regridded
zsmax_da.name = 'zsmax'
attr_da.name = 'attr'
zsmax_da = zsmax_da.drop_vars(['tc_id', 'scenario'])
attr_da = attr_da.drop_vars(['tc_id', 'scenario'])
zsmax_da.rio.write_crs(proj_crs, inplace=True)
attr_da.rio.write_crs(proj_crs, inplace=True)

rda_zsmax = zsmax_da
rda_attr = attr_da
print(wb_mask.rio.crs)
print(rda_zsmax.rio.crs)

print('Masking and calculating...')
# Mask out the water body grid cells (wb cell == 1)
mask = (wb_mask != 1)
zsmax_masked = rda_zsmax.where(mask)
# Mask out water levels below the ground elevation
mask = (zsmax_masked > elevation_da)
zsmax_masked = zsmax_masked.where(mask)
zsmax_masked.rio.write_crs(proj_crs, inplace=True)

# Calculate the depth above the ground
# Mask out depths smaller than the selected threshold
# Mask out the really large depths -- quarries or from model edge along the coastline
hmax = (zsmax_masked - elevation_da)
hmax.rio.write_crs(proj_crs, inplace=True)
mask = (hmax > hmin) & (hmax <= 10)
hmax_masked = hmax.where(mask)
hmax_masked.rio.write_crs(proj_crs, inplace=True)

# Mask out the attribution code data array
attr_masked = rda_attr.where(mask).astype(dtype='int8')
attr_masked.rio.write_crs(proj_crs, inplace=True)

# Runoff
mask = (attr_masked == 3)
hmax_runoff = hmax_masked.where(mask)
# Coastal
mask = (attr_masked == 1)
hmax_coastal = hmax_masked.where(mask)
# Compound
mask = attr_masked.isin([2,4])
hmax_compound = hmax_masked.where(mask)

# LOAD boundary condition data
boundary_condition_dir = rf'.\03_MODEL_RUNS\ncep_runs\completed_runs\TC_{str(target_tc_id).zfill(4)}\sfincs_bc_inputs'
precip_filepath = os.path.join(boundary_condition_dir, 'precip_2d.nc')
precip = sfincs_mod.data_catalog.get_rasterdataset(precip_filepath)
precip = precip.where(precip.notnull())
cumprecip = precip.sum(dim='time')

wind_filepath = os.path.join(boundary_condition_dir, 'wind_2d.nc')
wind = sfincs_mod.data_catalog.get_rasterdataset(wind_filepath)
wind['wdspd'] = np.sqrt((wind['eastward_wind']**2) + (wind['northward_wind']**2))
maxwind = wind['wdspd'].max(dim='time')

# Load layers - run once because it takes a while...
coastal_wb = sfincs_mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(32617)
coastal_wb_clip = coastal_wb.clip(sfincs_mod.region)

major_rivers = sfincs_mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
major_rivers = major_rivers.to_crs(32617)
major_rivers_clip = major_rivers.clip(sfincs_mod.region)

nc_major_rivers = sfincs_mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(32617)
nc_major_rivers_clip = nc_major_rivers.clip(sfincs_mod.region)

filtered_gdf = tc_tracks_polyline[tc_tracks_polyline['tc_id'].isin([target_tc_id])]
filtered_gdf = filtered_gdf.to_crs(proj_crs)
dataplot = [cumprecip, maxwind, hmax_masked, hmax_runoff, hmax_coastal, hmax_compound]
titlesplot = ['(a) Total Rainfall (mm)', '(b) Peak Wind Speed (m/s)','(c) Total Depth (m)',
              '(d) Runoff Depth (m)','(e) Coastal Depth (m)', '(f) Compound Depth (m)']
ckwargs_list = [dict(vmin=0, vmax=500, cmap='turbo'),
                dict(vmin=0, vmax=50, cmap='nipy_spectral'),
                dict(vmin=0.05, vmax=5, cmap='Blues')]


# Load CRS stuff for plotting
region = sfincs_mod.region
dem = sfincs_mod.grid['dep']
wkt = dem.raster.crs.to_wkt()
utm_zone = dem.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7, 8),
                        subplot_kw={'projection': utm},
                        sharex=True, sharey=True)
axs = axs.flatten()
for i in range(len(axs)):
    ax = axs[i]
    data = dataplot[i]
    if i > 1:
        ckwargs = ckwargs_list[2]
    else:
        ckwargs = ckwargs_list[i]

    dp = data.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
    if i > 1:
        region.plot(ax=ax, color='darkgrey', edgecolor='none', zorder=0, alpha=0.8)
        coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=0.5)
    else:
        major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=1)
        coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=1)

    major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=0.5)
    nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=0.5)
    region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=1, alpha=1)
    filtered_gdf.geometry.plot(ax=ax, color='black', linewidth=1.5, zorder=2)

    # Set figure extents
    minx, miny, maxx, maxy = region.buffer(10*3).total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    ax.set_title(titlesplot[i], fontsize=9)

    # Add a small inset colorbar — vertical on the right of each plot
    cax = inset_axes(ax,
                     width="6%",  # 3% of parent width
                     height="60%",  # 40% of parent height
                     loc='lower left',
                     bbox_to_anchor=(1.01, 0.1, 1, 1),  # (x0, y0, width, height)
                     bbox_transform=ax.transAxes,
                     borderpad=0,
                     )
    cbar = fig.colorbar(dp, cax=cax, orientation='vertical', extend='max')
    cbar.ax.tick_params(labelsize=9)

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.margins(x=0, y=0)
plt.savefig(
    rf'sfincs_inputs_outputs_{target_tc_id}.jpg',
    bbox_inches='tight', dpi=300)
plt.close()

