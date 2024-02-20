import os
import glob
import hydromt
import rioxarray
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
from shapely.geometry import box

# Filepath to data catalog yml
yml = os.path.join(r'Z:\users\lelise\data\data_catalog.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\design_storms')
rps = [2]
for rp in rps:
    mod = SfincsModel(root=f'nbll_40m_sbg3m_v3_eff25_P{rp}yr', mode='r', data_libs=yml)
    mod.read_results()

    tmax = mod.results['tmax'].max(dim='timemax')
    mask = np.isnat(tmax.values)  # mask out the NaT values
    tmax = tmax.astype(float)  # convert timedelta64[ns] to float
    tmax = tmax.where(~mask, np.nan)
    tmax = tmax / (3.6 * 10 ** 12)  # convert nanoseconds to hours
    tmax = xr.where(tmax >= 1, tmax, np.nan)
    tmax.raster.set_crs(mod.crs)
    tmax.raster.set_nodata(np.nan)
    tmax.raster.to_raster(join(mod.root, 'gis', f'nbll_P{rp}yr_tmax_hours_gridRes40m.tif'))

    fig, ax = mod.plot_basemap(
        fn_out=None,
        variable=tmax,
        plot_bounds=False,
        plot_geoms=False,
        bmap="sat",
        zoomlevel=11,
        vmin=0.04,
        vmax=22,
        cbar_kwargs={"shrink": 0.8, "anchor": (0, 0)},
    )
    ax.set_title(f"Time of Inundation > 0.1m (hours)")
    plt.savefig(join(mod.root, 'figs', f'P{rp}yr_tmax_hours.png'), dpi=225, bbox_inches="tight")
    plt.close()

    # Downscale Floodmaps
    dep_sbg = cat.get_rasterdataset(hydromt.data_catalog.join(
        r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\nbll_40m_sbg3m_v3_eff25',
        "subgrid",
        "dep_subgrid.tif"))

    hmax_sbg = utils.downscale_floodmap(
        zsmax=mod.results['zsmax'].max(dim='timemax'),
        dep=dep_sbg,
        hmin=0.1,
        gdf_mask=None,
        reproj_method='bilinear',
        floodmap_fn=join(mod.root, 'gis', f"nbll_P{rp}yr_max_depth_gridRes40m_sbg3m.tif")
    )

    dep = mod.grid['dep']
    hmax = utils.downscale_floodmap(
        zsmax=mod.results['zsmax'].max(dim='timemax'),
        dep=dep,
        hmin=0.1,
        gdf_mask=None,
        reproj_method='bilinear',
        floodmap_fn=join(mod.root, 'gis', f"nbll_P{rp}yr_max_depth_gridRes40m.tif")
    )
