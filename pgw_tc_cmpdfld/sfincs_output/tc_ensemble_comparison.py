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
import matplotlib.gridspec as gridspec


def compute_waterlevel_difference(da, scen_base, scen_keys=None, output_dir=None, output_tif=False):
    # Computer the difference in water level for compound compared to maximum single driver
    da_single_max = da.sel(run=scen_keys).max('run')
    da1 = (da.sel(run=scen_base) - da_single_max).compute()
    da1.name = 'diff. in waterlevel\ncompound - max. single driver'
    da1.attrs.update(unit='m')
    if output_tif is True:
        da1.rio.to_raster(os.path.join(output_dir, 'peak_depth_compound_minus_single_driver_sbg.tif'))
    return da1


yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models')
out_dir = os.path.join(os.getcwd(), '00_analysis', 'floodmaps')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scenarios = [
    'flor_ensmean_present',
    'flor_ens1_present',
    'flor_ens2_present',
    'flor_ens3_present',
    'flor_ens4_present',
    'flor_ens5_present',
    'flor_ens6_present',
    'flor_ens7_present'
]
scenarios_keys = [
    'Ensmean',
    'Ens1',
    'Ens2',
    'Ens3',
    'Ens4',
    'Ens5',
    'Ens6',
    'Ens7'
]


depfile=hydromt.data_catalog.join('flor_ensmean_present', "gis", "dep.tif")
da_list = []
hmin = 0.10
for scen in scenarios:
    # Read max water levels from model
    sfincs_mod = SfincsModel(root=scen, mode='r', data_libs=yml)
    sfincs_mod.read_results(fn_map=os.path.join(scen, 'sfincs_map.nc'),
                            fn_his=os.path.join(scen, 'sfincs_his.nc'))
    zsmax = sfincs_mod.results["zsmax"].max(dim='timemax')

    # Downscale results to get depth
    hmax = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=cat.get_rasterdataset(depfile),
        hmin=hmin,
        gdf_mask=None,
        reproj_method='bilinear',
        floodmap_fn=None)
    da_list.append(hmax)

da = xr.concat(da_list, dim='run')
da['run'] = xr.IndexVariable('run', scenarios_keys)

# Calculate mean depth across ensembles
da_ensemble_mean = da.sel(run=['Ens1', 'Ens2', 'Ens3', 'Ens4', 'Ens5', 'Ens6', 'Ens7']).mean('run')
diff = (da.sel(run=['Ensmean']) - da_ensemble_mean).compute()

# Calculate max depth across ensembles
da_ensemble_max = da.sel(run=['Ens1', 'Ens2', 'Ens3', 'Ens4', 'Ens5', 'Ens6', 'Ens7']).max('run')
diff = (da.sel(run=['Ensmean']) - da_ensemble_max).compute()

# Plotting info
mod = SfincsModel(scenarios[0], mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

plot_diff = True
if plot_diff is True:
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(6.5, 5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained')

    ckwargs = dict(cmap='seismic', vmin=-2, vmax=2)
    cs = diff.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=2)
    ax.set_title('')

    minx, miny, maxx, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_extent(extent, crs=utm)
    mod.region.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)

    # Colorbar - Precip
    label = 'Water Level Difference (m)\nEnsmean minus Max of ensembles'
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.01, pos0.y0 + pos0.height * 0.1, 0.025, pos0.height * 0.9])
    cbar = fig.colorbar(cs,
                        cax=cax,
                        orientation='vertical',
                        label=label,
                        extend='both')

    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'ensemble_diff_max.png'),
                bbox_inches='tight',
                dpi=255)
    plt.close()
