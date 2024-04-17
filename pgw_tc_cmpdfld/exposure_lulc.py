#!/usr/bin/env python
# coding: utf-8

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
from xhistogram.xarray import histogram


def get_flooded_landcover(mod, lulc_file, dst_crs, hmin=0.05, mask=None, dep_file=None):
    if dep_file is None:
        dep = mod.grid['dep']
    else:
        dep = cat.get_rasterdataset(dep_file)

    if mask is None:
        mask = mod.region
    else:
        mask = mask.to_crs(dst_crs)

    # Downscaled floodmap
    hmax = utils.downscale_floodmap(zsmax=mod.results["zsmax"].max(dim='timemax'),
                                    dep=dep, hmin=hmin, gdf_mask=mask, reproj_method='bilinear', floodmap_fn=None)
    minx, miny, maxx, maxy = mask.total_bounds
    hmax = hmax.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    # Get, clip, mask LULC data
    lulc_da = cat.get_rasterdataset(lulc_file, bbox=mask.to_crs(4326).total_bounds)
    lulc_da = lulc_da.raster.reproject(dst_crs=dst_crs, dst_nodata=255)  # reproject the raster to model crs
    lulc_da = lulc_da.interp(x=hmax.x, y=hmax.y, method='nearest')  # interpolate to the flood depth resolution
    lulc_da = lulc_da.where(hmax > hmin)  # mask areas that were flooded

    return lulc_da


# Load the model the results
yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models')
mask = cat.get_geodataframe(r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\model_data\model_domain_HUC10.shp')
dep_file = r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\flo_hindcast_v6_200m_LPD2m_avgN_20msbg\subgrid' \
          r'\dep_subgrid.tif'

iclus2050 = r'Z:\users\lelise\data\lulc\ICLUS\ICLUS_v2_1_1_land_use_conus_2030_ssp5_rcp85_hadgem2_es.tif'

out_dir = os.path.join(os.getcwd(), '00_analysis', 'floodmaps2')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

mod_root = [r'present_florence\ensmean\flor_ensmean_present',
            r'future_florence\future_florence_ensmean']

# NLCD Mapping
# mapping_nlcd = cat.get_dataframe(r'Z:\users\lelise\data\lulc\nlcd_lulc_mapping.csv')
# bins = [int(i) for i in mapping_nlcd['nlcd']] + [255]

# ICLUS
mapping_iclus = cat.get_dataframe(r'Z:\users\lelise\data\lulc\iclus_lulc_mapping.csv')
bins = [int(i) for i in mapping_iclus['code']] + [255]

###
res = 20
mapping = mapping_iclus

lulc_list = []
for root in mod_root:
    m = os.path.basename(root)
    scenarios = [m, (m + '_coastal'), (m + '_runoff')]
    for scen in scenarios:
        mod = SfincsModel(root=os.path.join(os.getcwd(), os.path.dirname(root), scen), mode='r', data_libs=yml)

        lulc = get_flooded_landcover(mod=mod, lulc_file=iclus2050, dst_crs=32617,
                                     hmin=0.05, mask=mask, dep_file=dep_file)
        #lulc_list.append(lulc)
        binned = histogram(lulc, bins=[np.array(bins)])
        mapping[scen] = binned.values

lulc_fld_area = mapping.copy()
exclude_col = ['code', 'roughness', 'reference description', 'reference', 'iclus description', 'nlcd_code']
lulc_fld_area[lulc_fld_area.columns.difference(exclude_col)] = lulc_fld_area[lulc_fld_area.columns.difference(exclude_col)] * (res * res) / (1000 ** 2)
#lulc_fld_area[lulc_fld_area.columns.difference(['land_type', 'nlcd'])] = lulc_fld_area[lulc_fld_area.columns.difference(['land_type', 'nlcd'])] * (res * res) / (1000 ** 2)
lulc_fld_area.to_csv(os.path.join(out_dir, 'mapping_20m_res_iclus2030.csv'))
#     # Downscale water level to subgrid
#     da = downscale_floodmaps(model_root=os.path.join(os.getcwd(), os.path.dirname(mod_root[counter])),
#                              fname_key='grid',
#                              scenarios=scenarios,
#                              depfile=grid_file,
#                              hmin=hmin,
#                              scen_keys=scenarios_keys,
#                              gdf_mask=None,
#                              output_folder=out_dir,
#                              output_nc=False,
#                              output_tif=True
#                              )
#     da['run'] = xr.IndexVariable('run', scenarios_keys)
#     # Calculate difference in water level
#     da1 = compute_waterlevel_difference(da=da,
#                                         scen_base='compound',
#                                         scen_keys=['coastal', 'runoff'],
#                                         output_dir=out_dir,
#                                         output_tif=False
#                                         )
#     compound_mask = da1 > hmin
#     coastal_mask = da.sel(run='coastal').fillna(0) > da.sel(run=['runoff']).fillna(0).max('run')
#     runoff_mask = da.sel(run='runoff').fillna(0) > da.sel(run=['coastal']).fillna(0).max('run')
#     assert ~np.logical_and(runoff_mask, coastal_mask).any()
#     da_c = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
#             + xr.where(runoff_mask, x=compound_mask + 3, y=0)
#             ).compute()
#     da_c.name = None
#
#     # Calculate the frequency of the drivers at the grid/subgrid resolution
#     da_cd = da_c.data
#     unique, counts = np.unique(da_cd, return_counts=True)
#     fld_area_by_driver.append(counts)
#
#     counter += 1
#
#
# please_plot = True
# mask = mask.to_crs(mod.crs)
# if please_plot is True:
#     wkt = mod.grid['dep'].raster.crs.to_wkt()
#     utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
#     utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
#     extent = np.array(mask.buffer(100).total_bounds)[[0, 2, 1, 3]]
#
#     fig, ax = plt.subplots(
#         nrows=1, ncols=1,
#         figsize=(4.5, 6),
#         # subplot_kw={'projection': utm},
#         tight_layout=True)
#
#     lulc_nb.plot.hist(ax=ax)
#
#     # Setup figure extents
#     # ax.set_extent(extent, crs=utm)
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\00_analysis\floodmaps2\test.png',
#                 dpi=225, bbox_inches="tight")
#     plt.close()
