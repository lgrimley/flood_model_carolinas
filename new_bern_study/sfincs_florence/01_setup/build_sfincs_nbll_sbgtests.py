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
cat_dir = '/projects/sfincs/data'
# cat_dir = r'Z:\users\lelise\data'
yml_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_Carolinas2 = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_NBLL = os.path.join(cat_dir, 'data_catalog_SFINCS_NewBernNC.yml')

# Working directory and model root
os.chdir('/projects/sfincs/SFINCS_NBLL')
root = 'nbll_model_v3'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_CONUS, yml_Carolinas, yml_Carolinas2, yml_NBLL])

mask_dataset = 'nbll_domain'
domain = cat.get_geodataframe(mask_dataset)
domain.to_crs(epsg=4326, inplace=True)

# Setup model region
mod.setup_region(region={'geom': cat[mask_dataset].path})
print('Setup model region')

# Setup grid
grid_res = 80
sbg_res = 3
mod.setup_grid_from_region(
    region={'geom': cat[mask_dataset].path},
    res=grid_res,
    crs='utm',
    rotated=False
)
_ = mod.plot_basemap(fn_out='region.png',
                     plot_region=True,
                     bmap='sat',
                     variable='dep',
                     plot_geoms=False
                     )
plt.close()
print('Setup Grid')

# Add topobathy data to grid
datasets_dep = [
    {"elevtn": "nc_RASbathy_Neuse", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_RASbathy_Coastal", 'reproj_method': 'bilinear'},

    {"elevtn": "nc_ChanNHDArea_RASbedInterp", 'reproj_method': 'bilinear'},

    {"elevtn": "nc_Chan5mWdth_RASbed_Neuse", 'reproj_method': 'bilinear'},

    {"elevtn": "nc_2m_DEM_USGS_CoNED_tiles", 'reproj_method': 'bilinear'},

    {"elevtn": "nc_2m_region3_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
    {"elevtn": "nc_2m_region4_tiles", 'reproj_method': 'bilinear', 'zmin': -1},

    {"elevtn": "Carolinas_10m_DEM_USGS_NED", 'reproj_method': 'bilinear', 'zmin': 5},

    {"elevtn": "nc_3m_DEM_NOAA_CUDEM", 'reproj_method': 'bilinear', 'zmax': 10},
    {"elevtn": "gebco"},
]

dep = mod.setup_dep(datasets_dep=datasets_dep, buffer_cells=0)

_ = mod.plot_basemap(fn_out='terrain.png', variable="dep", bmap="sat", zoomlevel=5)
plt.close()
print('Done with Topography')
mod.write_grid()

# Setup Mask
mod.setup_mask_active(mask=mask_dataset, reset_mask=True)
print('Done with setting up mask')
mod.write()
#
# Identify boundary cells
t = cat.get_geodataframe('carolinas_coastal_wb').to_crs(mod.crs)
mod.setup_mask_bounds(btype='waterlevel',
                      include_mask=t,
                      connectivity=8,
                      zmin=-50,
                      zmax=0,
                      reset_bounds=True)
t = cat.get_geodataframe('carolinas_major_rivers').to_crs(mod.crs)
mod.setup_mask_bounds(btype='waterlevel',
                      include_mask=t,
                      reset_bounds=False
                      )


mod.setup_mask_bounds(btype='outflow',
                      include_mask='nbll_bc_outflow',
                      reset_bounds=True)
_ = mod.plot_basemap(fn_out='mask.png', variable="msk", plot_bounds=True, bmap="sat", zoomlevel=12)
plt.close()
mod.write_grid()
print('Done with setting up bounds')

# Setup Model Manning's File
lulc = mod.data_catalog.get_rasterdataset('nlcd_2016', geom=mod.region)

# rasterize the manning value of gdf to the model grid - rivers and coastal water bodies
nhd_area = mod.data_catalog.get_geodataframe("carolinas_nhd_area_rivers", geom=mod.region).to_crs(mod.crs)
nhd_area["manning"] = 0.035
nhd_area_manning = lulc.raster.rasterize(nhd_area, "manning", nodata=np.nan, all_touched=False)

rivers = mod.data_catalog.get_geodataframe("fris_stream_cntrline", geom=mod.region).to_crs(mod.crs)
rivers["manning"] = 0.045
rivers_manning = lulc.raster.rasterize(rivers, "manning", nodata=np.nan, all_touched=False)

coastal_wb = mod.data_catalog.get_geodataframe("carolinas_coastal_wb", geom=mod.region).to_crs(mod.crs)
coastal_wb["manning"] = 0.022
coastal_wb_manning = lulc.raster.rasterize(coastal_wb, "manning", nodata=np.nan, all_touched=False)

datasets_rgh = [
    {"manning": coastal_wb_manning},
    {"manning": nhd_area_manning},
    {"manning": rivers_manning},
    {"lulc": "nlcd_2016",
     'reclass_table': os.path.join(cat_dir, 'lulc/nlcd/nlcd_mapping_max.csv')}]

mod.setup_manning_roughness(datasets_rgh=datasets_rgh)
_ = mod.plot_basemap(fn_out='mannings.png', variable="manning", plot_bounds=False, bmap="sat", zoomlevel=12)
plt.close()
print('Done with setting up mannings roughness')

# Setup Curve Number Infiltration
mod.setup_cn_infiltration(cn='gcn250',
                          antecedent_moisture='avg')
_ = mod.plot_basemap(fn_out='scs_curvenumber.png', variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
plt.close()
mod.write()
print('Done with setting up CN infiltration without recovery')

hsg = mod.data_catalog.get_rasterdataset('gNATSGO_hsg_conus', geom=mod.region)
ksat = mod.data_catalog.get_rasterdataset('gNATSGO_ksat_DCP_0to20cm_carolinas', geom=mod.region)

mod.setup_cn_infiltration_with_ks(lulc=lulc,
                                  hsg=hsg,
                                  ksat=ksat,
                                  reclass_table=os.path.join(cat_dir, 'soil/surrgo/CN_Table_HSG_NLCD.csv'),
                                  effective=0.50,
                                  block_size=2000)
print('Done with setting up CN infiltration with recovery')
mod.write()

# Updating config
mod.setup_config(
    **{
        'crsgeo': mod.crs.to_epsg(),
        "tref": "20180907 000000",
        "tstart": "20180907 000000",
        "tstop": "20180930 000000",

        'dtrstout': '259200',
        'dtout': '3600',
        'dthisout': '1800',
        'tspinup': '86400',

        'advection': '1',
        'alpha': '0.5',
        'theta': '1',
        'huthresh': '0.05',
        'viscosity': '1',

        'min_lev_hmax': '-20',
        'zsini': '0.25',
        'stopdepth': '100',

        'rhoa': '1.25',
        'cd_nr': '0',
        'cdnrb': '3',
        'cdwnd': '0 28 50',
        'cdval': '0.0010 0.00250 0.0025',

        'storetwet': '1',
        'twet_threshold': '0.1',
        # 'storevel': '1',
        # 'storemeteo': '1',
        # 'storecumprcp': '1',
        # 'storevelmax': '1',
        # 'storemaxwind': '1',
    }
)
print(mod.config)
mod.write_config(config_fn='sfincs.inp')

# Setup Structures
mod.setup_structures('nbll_weirs',
                     stype='thd',
                     dep=None,
                     buffer=None,
                     merge=False
                     )
data = mod.geoms['thd']
data.to_file(os.path.join(os.getcwd(), root, 'gis', 'thd.shp'))
print('Setup structures')

# Setup water level forcing
mod.setup_waterlevel_forcing(geodataset='adcirc_waterlevel_florence_extended',
                             offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=2000,
                             merge=False)
mod.write_forcing(data_vars='bzs')
gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# Setup water level forcing
mod.setup_waterlevel_forcing(geodataset='usgs_waterlevel_florence',
                             timeseries=None,
                             locations=None,
                             buffer=1000,
                             merge=True
                             )
mod.write_forcing(data_vars='bzs')
gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# # Setup discharge forcing
# mod.setup_discharge_forcing(geodataset='harvey_bc_discharge',
#                             merge=False)
# mod.write_forcing(data_vars='dis')
# gdf_locs = mod.forcing['dis'].vector.to_gdf()
# gdf_locs['name'] = mod.forcing['dis'].index.values
# gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
# print('Write dis')

# Setup gridded precipitation forcing
mod.setup_precip_forcing_from_grid(precip='mrms_tc_florence', aggregate=False)
mod.write_forcing(data_vars='precip')
print('Write precip')

# Write wind forcing
mod.setup_wind_forcing_from_grid(wind='owi_florence_winds')
mod.write_forcing(data_vars='wind')
print('Writing wind')
mod.write_forcing()

_ = mod.plot_forcing(fn_out="forcing.png")
plt.close()
print('Plot forcing')

mod.write_forcing()
mod.write()
_ = mod.plot_forcing(fn_out="forcing.png")
plt.close()
print('Plot forcing')

# # Setup observation points
# obs_datasets = ['usgs_waterlevel_florence',
#                 #'noaa_waterlevel_florence',
#                 'usgs_rapid_deployment_waterlevel_florence',
#                 'ncem_waterlevel_florence'
#                 ]
# for i in range(len(obs_datasets)):
#     data = cat.get_geodataset(data_like=obs_datasets[i],
#                               geom=domain,
#                               buffer=0,
#                               variables=["waterlevel"],
#                               time_tuple=mod.get_model_time()
#                               )
#     if 'geometry' in list(data.coords):
#         df = gpd.GeoDataFrame(data.index.values,
#                               geometry=data.geometry.values,
#                               crs=4326)
#     else:
#         df = gpd.GeoDataFrame(data.index.values,
#                               geometry=gpd.points_from_xy(x=data.x.values,
#                                                           y=data.y.values),
#                               crs=4326)
#     df.columns = ['site_no', 'geometry']
#     if i == 0:
#         mod.setup_observation_points(locations=df, merge=False)
#     else:
#         mod.setup_observation_points(locations=df, merge=True)
#
# data = mod.geoms['obs']
# data.to_file(os.path.join(mod.root, 'gis', 'obs.shp'))
# mod.write_geoms(data_vars='obs')
# print('Done setting up point observations')

# # Setup observation cross-sections
# mod.setup_observation_lines('discharge_crs_carolinas')
# data = mod.geoms['crs']
# data.to_file(os.path.join(os.getcwd(), root, 'gis', 'crs.shp'))
# print('Done setting up cross-section observations')

# Write and plot model
_ = mod.plot_basemap(fn_out="basemap.png", bmap="sat")
plt.close()
mod.write()

# Add Setup Model Subgrid
print('Writing subgrid...')
mod.setup_subgrid(
    datasets_dep=datasets_dep,
    datasets_rgh=datasets_rgh,
    nr_subgrid_pixels=int(grid_res / sbg_res),
    nbins=15,
    write_dep_tif=True,
    write_man_tif=False,
)

mod.write_subgrid()
print('Done with subgrid')
mod.write()
