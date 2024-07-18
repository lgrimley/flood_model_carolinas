import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter

# THIS SCRIPT COMPILES ALL THE FLOOD DEPTHS FOR EACH SCENARIO (E.G., STORM, CLIMATE) AND CREATES A DISTRIBUTION
# TO EXTRACT DEPTH STATISTICS. THIS CAN BE COMPARED TO TAKING THE MEAN ACROSS THE ENSEMBLE AT EACH CELL TO GET A
# MEAN DEPTH SURFACE... THEN FITTING THE DISTRIBUTION TO THE DEPTHS.

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis')

depfile = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\subgrid\dep.tif'

root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\AGU2023\future_florence' \
       r'\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')

da_zsmax = xr.open_dataarray(os.path.join(os.getcwd(), 'zsmax', f'pgw_zsmax.nc'))
da_zsmax_classified = xr.open_dataarray(os.path.join(os.getcwd(), 'driver_analysis', f'pgw_drivers_classified_all.nc'))

# Loop through the zsmax and calculate the depths for the water levels for each attributing process
runs = list(da_zsmax.run.values)
runs = list(filter(lambda x: 'flor_pres' in x, runs))
da_class = da_zsmax_classified.sel(run='flor_pres')

# Consider only areas where compound flooding occurred
mask = ((da_class == 2) | (da_class == 4))

# Get water depths in these areas only
hmax_das = []
for run in runs:
    print(run)
    zsmax = da_zsmax.sel(run=run)
    hmax = utils.downscale_floodmap(
        zsmax=zsmax.where(mask),
        dep=mod.data_catalog.get_rasterdataset(depfile),
        hmin=0.05,
        gdf_mask=None,
        reproj_method='bilinear'
    )
    hmax_das.append(hmax)

# Add the depths for coastal and runoff 
summed_processes = (hmax_das[0] + hmax_das[2]).compute()
hmax_das.append(summed_processes)
hmax_das = xr.concat(hmax_das, dim='run')
runs.append('summed_processes')
hmax_das['run'] = xr.IndexVariable('run', runs)

hmax_df = pd.DataFrame()
for run in hmax_das.run.values.tolist():
    # Get depths and combine into single pandas dataframe
    hmax = hmax_das.sel(run=run)
    df = hmax.to_dataframe().dropna(how='any', axis=0)
    df = pd.DataFrame(df['hmax'])
    df.reset_index(inplace=True)
    df.drop(columns=df.columns.difference(['hmax']), inplace=True)
    hmax_df = pd.concat([hmax_df, df], ignore_index=True, axis=1)
hmax_df.columns = hmax_das.run.values.tolist()
mean_depth = hmax_df.mean()

# Create plot
plot_depth_hist = False
if plot_depth_hist is True:
    nbins = 50
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(nrows=len(hmax_df.columns),
                             figsize=(4, 5),
                             ncols=1,
                             tight_layout=True, layout='constrained',
                             sharex=True,
                             sharey=True
                             )
    for i in range(len(hmax_df.columns)):
        col = hmax_df.columns[i]
        data, bins, _ = axes[i].hist(hmax_df[col],
                                     bins=nbins,
                                     range=[0, 15],
                                     density=True,
                                     histtype='stepfilled',
                                     alpha=0.8
                                     )
        axes[i].axvline(x=mean_depth[col], color='red')
        axes[i].set_title(col)

    plt.xlabel('Flood Depths (m)')
    plt.suptitle('Depths in Compound Areas')
    plt.subplots_adjust(wspace=0.02, hspace=0.4, top=0.90)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter1\flor_depths_in_compoundAreas.png'),
                bbox_inches='tight', dpi=255)
    plt.close()

''' PART 1 - Determine damage status using NFIP claims/policy data at each structure'''
# Read in area of interest shapefile and project
studyarea_gdf = mod.region.to_crs(epsg=32617)

# Read in structures information and clip to the study area
nc_buildings = gpd.read_file(r'Z:\users\lelise\geospatial\flood_damage\included_data.gdb',
                             layer='buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['STATE'] = 'NC'
b1 = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of NC Buildings in Study Area:', str(len(nc_buildings)))

# Load SC buildings from NSI
sc_buildings = gpd.read_file(r'Z:\users\lelise\geospatial\infrastructure\nsi_2022_45.gpkg',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
sc_buildings['STATE'] = 'SC'
b2 = sc_buildings.drop(sc_buildings.columns[~sc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of SC Buildings in Study Area:', str(len(sc_buildings)))

buildings = pd.concat(objs=[b1, b2],
                      axis=0,
                      ignore_index=True)

# Extract depth at buildings
gdf = buildings
hmin = 0.05
hmax_df = pd.DataFrame()
for run in hmax_das.run.values.tolist():
    # Get depths and combine into single pandas dataframe
    hmax = hmax_das.sel(run=run)
    hmax_at_buildings = hmax.sel(x=gdf['geometry'].x.to_xarray(),
                                 y=gdf['geometry'].y.to_xarray(),
                                 method='nearest').values
    gdf[run] = hmax_at_buildings.transpose()

gdf_clean = gdf.copy()
gdf_clean.drop(['STATE'], inplace=True, axis=1)
gdf_clean = gdf_clean.dropna(axis=0, how='all', subset=hmax_das.run.values.tolist())
gdf_clean.set_index('geometry', drop=True, inplace=True)
gdf_clean['summed_processes'] = gdf_clean['flor_pres_coastal'].fillna(0) + gdf_clean['flor_pres_runoff'].fillna(0)

mean_depth = gdf_clean.mean()
median_depth = gdf_clean.median()
std_depth = gdf_clean.std()

# Create plot
plot_depth_hist = True
if plot_depth_hist is True:
    nbins = 50
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(nrows=len(gdf_clean.columns),
                             figsize=(4, 5),
                             ncols=1,
                             tight_layout=True, layout='constrained',
                             sharex=True,
                             sharey=True
                             )
    for i in range(len(gdf_clean.columns)):
        col = gdf_clean.columns[i]
        d = gdf_clean[col]
        d.dropna(axis=0, how='any', inplace=True)
        data, bins, _ = axes[i].hist(d,
                                     bins=nbins,
                                     range=[0, 7],
                                     density=True,
                                     histtype='stepfilled',
                                     alpha=0.8
                                     )
        axes[i].axvline(x=mean_depth[col], color='red')
        axes[i].axvline(x=median_depth[col], color='purple')
        axes[i].set_title(f'{col} (n={len(d)})')

    plt.xlabel('Flood Depths (m)')
    plt.suptitle('Building Depths in Compound Areas')
    plt.subplots_adjust(wspace=0.02, hspace=0.4, top=0.90)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter1\flor_depths_in_compoundAreas_atBuildings.png'),
                bbox_inches='tight', dpi=255)
    plt.close()
