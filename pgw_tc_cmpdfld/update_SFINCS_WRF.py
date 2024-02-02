import os
import datetime
import hydromt
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Filepath to data catalog yml
# cat_dir = '/projects/sfincs/data'
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
os.chdir(r'Z:\users\lelise\projects\HCFCD\sfincs_models\03_for_TAMU\DesignStorms\mod_v2')
# os.chdir('/projects/sfincs/')
root = 'hcfcd_100m_sbg5m_v2_fut_500yr'
mod = SfincsModel(root=root, mode='r+', data_libs=yml)

mod.update(r'Z:\users\lelise\projects\Carolinas\Chapter2\sfincs_models\matt_ensmean_future_110cm')

# mask_dataset = 'enc_domain_HUC6_clipped'
# domain = cat.get_geodataframe(mask_dataset)
# domain.to_crs(epsg=4326, inplace=True)
#
# # Setup model region
# mod.setup_region(region={'geom': cat[mask_dataset].path})
# print('Setup model region')
#
# # Setup grid
# grid_res = 200
# sbg_res = 5
# mod.setup_grid_from_region(
#     region={'geom': cat[mask_dataset].path},
#     res=grid_res,
#     crs='utm',
#     rotated=False
# )
# _ = mod.plot_basemap(fn_out='region.png',
#                      plot_region=True,
#                      bmap='sat',
#                      # zoomlevel=15,
#                      variable='dep',
#                      plot_geoms=False
#                      )
# plt.close()
# print('Setup Grid')
#
# # Add topobathy data to grid
# datasets_dep = [
#     {"elevtn": "neuse_bathy_tiles", 'reproj_method': 'bilinear'},
#     {"elevtn": "coastal_bathy_tiles", 'reproj_method': 'bilinear'},
#     {"elevtn": "tar_bathy_tiles_v2", 'reproj_method': 'bilinear'},
#     {"elevtn": "peedee_bathy_tiles", 'reproj_method': 'bilinear'},
#     {"elevtn": "capefear_bathy_tiles", 'reproj_method': 'bilinear'},
#
#     {"elevtn": "nhd_area_interp", 'reproj_method': 'bilinear'},
#     {"elevtn": "nhd_area_peedee_2mburn", 'reproj_method': 'bilinear'},
#
#     {"elevtn": "capefear_5m_rect_channel", 'reproj_method': 'bilinear'},
#     {"elevtn": "lower_peedee_5m_rect_channel", 'reproj_method': 'bilinear'},
#     {"elevtn": "neuse_5m_rect_channel", 'reproj_method': 'bilinear'},
#     {"elevtn": "pamlico_5m_rect_channel", 'reproj_method': 'bilinear'},
#     {"elevtn": "onslow_bay_5m_rect_channel", 'reproj_method': 'bilinear'},
#
#     {"elevtn": "coned_sc_2m_tiles", 'reproj_method': 'bilinear'},
#     {"elevtn": "coned_nc_2m_tiles", 'reproj_method': 'bilinear'},
#
#     {"elevtn": "nc_2m_region3_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
#     {"elevtn": "nc_2m_region4_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
#
#     {"elevtn": "ned_2m_sc_savannahPeeDee_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
#     {"elevtn": "ned_2m_sc_georgetown", 'reproj_method': 'bilinear'},
#     {"elevtn": "ned_2m_sc_williamsburg", 'reproj_method': 'bilinear'},
#     {"elevtn": "ned_2m_sc_eastCentral", 'reproj_method': 'bilinear'},
#     {"elevtn": "ned_2m_sc_berkeley", 'reproj_method': 'bilinear'},
#     {"elevtn": "ned_2m_sc_charleston", 'reproj_method': 'bilinear'},
#
#     {"elevtn": "ned_10m_carolinas", 'reproj_method': 'bilinear', 'zmin': 5},
#
#     {"elevtn": "cudem_nc", 'reproj_method': 'bilinear', 'zmax': 10},
#     {"elevtn": "cudem_southeast", 'reproj_method': 'bilinear', 'zmax': 10},
#     {"elevtn": "gebco"},
# ]
#
# dep = mod.setup_dep(datasets_dep=datasets_dep)
#
# _ = mod.plot_basemap(fn_out='terrain.png', variable="dep", bmap="sat", zoomlevel=5)
# plt.close()
# print('Done with Topography')
# mod.write_grid()
#
# # Setup Mask
# mod.setup_mask_active(mask=mask_dataset,
#                       zmin=-30,
#                       reset_mask=True)
# print('Done with setting up mask')
#
# # Identify boundary cells
# mod.setup_mask_bounds(btype='waterlevel',
#                       include_mask='carolinas_coastal_wb',
#                       connectivity=8,
#                       zmin=-50,
#                       zmax=0,
#                       reset_bounds=True)
#
# mod.setup_mask_bounds(btype='outflow',
#                       include_mask='outflow_bc',
#                       reset_bounds=True)
# _ = mod.plot_basemap(fn_out='mask.png', variable="msk", plot_bounds=True, bmap="sat", zoomlevel=12)
# plt.close()
# mod.write_grid()
# print('Done with setting up bounds')
#
# # Setup Model Manning's File
# lulc = mod.data_catalog.get_rasterdataset('nlcd_2016', geom=mod.region)
#
# # rasterize the manning value of gdf to the model grid - rivers and coastal water bodies
# nhd_area = mod.data_catalog.get_geodataframe("carolinas_nhd_area_rivers", geom=mod.region).to_crs(mod.crs)
# nhd_area["manning"] = 0.030
# nhd_area_manning = lulc.raster.rasterize(nhd_area, "manning", nodata=np.nan, all_touched=False)
#
# rivers = mod.data_catalog.get_geodataframe("fris_stream_cntrline", geom=mod.region).to_crs(mod.crs)
# rivers["manning"] = 0.045
# rivers_manning = lulc.raster.rasterize(rivers, "manning", nodata=np.nan, all_touched=False)
#
# coastal_wb = mod.data_catalog.get_geodataframe("carolinas_coastal_wb", geom=mod.region).to_crs(mod.crs)
# coastal_wb["manning"] = 0.022
# coastal_wb_manning = lulc.raster.rasterize(coastal_wb, "manning", nodata=np.nan, all_touched=False)
#
# datasets_rgh = [
#     {"manning": coastal_wb_manning},
#     {"manning": nhd_area_manning},
#     {"manning": rivers_manning},
#     {"lulc": "nlcd_2016",
#      'reclass_table': os.path.join(cat_dir, 'lulc/nlcd/nlcd_mapping_mean.csv')}]
#
# mod.setup_manning_roughness(datasets_rgh=datasets_rgh)
# _ = mod.plot_basemap(fn_out='mannings.png', variable="manning", plot_bounds=False, bmap="sat", zoomlevel=12)
# plt.close()
# print('Done with setting up mannings roughness')
#
# # Setup Curve Number Infiltration
# mod.setup_cn_infiltration(cn='gcn250',
#                           antecedent_moisture='avg')
# _ = mod.plot_basemap(fn_out='scs_curvenumber.png', variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
# plt.close()
# mod.write()
# print('Write model just in case')

# Updating config

mod.config.update(
    **{
        "tref": "20161007 000000",
        "tstart": "20161007 000000",
        "tstop": "20161015 000000",
        "advection": 1,
    }
)
print(mod.config)
mod.write_config(config_fn='sfincs.inp')

# # Setup Structures
# mod.setup_structures('levees_carolinas',
#                      stype='weir',
#                      dep=None,
#                      buffer=None,
#                      merge=False,
#                      dz=0)
# data = mod.geoms['weir']
# data.to_file(os.path.join(os.getcwd(), root, 'gis', 'weir.shp'))
# print('Setup structures')

# Setup water level forcing


mod.setup_waterlevel_forcing(geodataset='adcirc_wrf_matt_ensmean_pgw_1.10m',
                             offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=500,
                             merge=False)
bzs = mod.forcing['bzs']
to_remove = []
for ind in range(len(bzs['index'])):
    zs = bzs[:, ind]
    if zs.max() > 10:
        t = zs['index'].data.item()
        to_remove.append(t)
        print(t)

cleaned_bcs = bzs.drop_sel(index=to_remove)
mod.setup_waterlevel_forcing(geodataset=cleaned_bcs, merge=False)
mod.write_forcing(data_vars='bzs')
gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# Setup discharge forcing
mod.setup_discharge_forcing(geodataset='usgs_discharge_matthew',
                            merge=False,
                            buffer=2000)
mod.write_forcing(data_vars='dis')
gdf_locs = mod.forcing['dis'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['dis'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
print('Write dis')

# Setup gridded precipitation forcing
mod.setup_precip_forcing_from_grid(precip='wrf_matt_ensmean_future', aggregate=False)
mod.write_forcing(data_vars='precip')
print('Write precip')

# Write wind forcing
mod.setup_wind_forcing_from_grid(wind='wrf_matt_ensmean_future')
mod.write_forcing(data_vars='wind')
print('Writing wind')

_ = mod.plot_forcing(fn_out='forcings_meteo.png', forcings=['precip'])
plt.close()

_ = mod.plot_forcing(fn_out='forcings_meteo.png', forcings=['precip_2d', 'wind_u', 'wind_v'])
plt.close()

mod.write_forcing()
mod.write()
