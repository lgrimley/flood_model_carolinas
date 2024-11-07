import os
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import rasterio.merge

import hydromt
import hydromt_sfincs
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils


# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')

# Script start
startTime = datetime.datetime.now()
print(f'Script started: {startTime}')

# Filepath to data catalog ymls
cat_dir = '/projects/sfincs/data'

# Load in the data catalogs needed for building the model
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_sfincs_Carolinas_Ch3 = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas_Ch3.yml')

# Setup working directory and model root, create and instance of a SFINCS model to write to
os.chdir('/projects/sfincs/syntheticTCs_cmpdfld')
root = 'ENC_200m_sbg5m_avgN_eff75'
mod = SfincsModel(root=root, mode='w+',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas, yml_sfincs_Carolinas_Ch3])
# This will have all the data from the multiple data catalogs included so easier to query
cat = mod.data_catalog

# Read in the model domain mask
mask_dataset = 'eastern_carolinas_domain_chapter3'
domain = cat.get_geodataframe(mask_dataset)
domain.to_crs(epsg=4326, inplace=True)
print(f'Domain active areas set using: {mask_dataset}')

# Setup model region
mod.setup_region(region={'geom': cat[mask_dataset].path})
print('Setup model region')

# Setup grid
grid_res = 200
sbg_res = 5
print(f'Model grid resolution = {grid_res}. Model subgrid resolution = {sbg_res}')
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
print('Setup grid and plot basemap')

# Add topobathy data to grid
# Set up a dictionary of elevation rasters to assign to the grid
datasets_dep = [
    # HEC-RAS Interpolation surface rasters
    {"elevtn": "nc_RASbathy_Neuse_tiles", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_RASbathy_Coastal_tiles", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_RASbathy_Tar_tiles", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_RASbathy_LowerPeeDee_tiles", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_RASbathy_CapeFear_tiles", 'reproj_method': 'bilinear'},

    # NHD Area rasterized and linear interpolation of point bathy data to this grid
    {"elevtn": "nc_ChanNHDArea_RASbedInterp", 'reproj_method': 'bilinear'},

    # NHD Area rasterized and a constant depth burned into the channel base on underlying DEM
    {"elevtn": "sc_LPD_bathy_2mburn", 'reproj_method': 'bilinear'},

    # FRIS stream centerlines rasterized to 5m channel. FRIS point data interpolated onto this raster.
    {"elevtn": "nc_Chan5mWdth_RASbed_CapeFear", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_Chan5mWdth_RASbed_LPD", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_Chan5mWdth_RASbed_Neuse", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_Chan5mWdth_RASbed_Pamlico", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_Chan5mWdth_RASbed_OnslowBay", 'reproj_method': 'bilinear'},

    # USGS CoNED data
    {"elevtn": "sc_2m_DEM_USGS_CoNED_tiles", 'reproj_method': 'bilinear'},
    {"elevtn": "nc_2m_DEM_USGS_CoNED_tiles", 'reproj_method': 'bilinear'},

    # NC State Lidar
    {"elevtn": "nc_2m_region3_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
    {"elevtn": "nc_2m_region4_tiles", 'reproj_method': 'bilinear', 'zmin': -1},

    # USGS NED data
    {"elevtn": "sc_2m_DEM_USGS_NED_SavannahPeeDee_tiles", 'reproj_method': 'bilinear', 'zmin': -1},
    {"elevtn": "sc_2m_DEM_USGS_NED_Georgetown", 'reproj_method': 'bilinear'},
    {"elevtn": "sc_2m_DEM_USGS_NED_Williamsburg", 'reproj_method': 'bilinear'},
    {"elevtn": "sc_2m_DEM_USGS_NED_EastCentral", 'reproj_method': 'bilinear'},
    {"elevtn": "sc_2m_DEM_USGS_NED_Berkeley", 'reproj_method': 'bilinear'},
    {"elevtn": "sc_2m_DEM_USGS_NED_Charleston", 'reproj_method': 'bilinear'},
    {"elevtn": "Carolinas_10m_DEM_USGS_NED", 'reproj_method': 'bilinear', 'zmin': 5},

    {"elevtn": "nc_3m_DEM_NOAA_CUDEM", 'reproj_method': 'bilinear', 'zmax': 10},
    {"elevtn": "SouthEast_3m_DEM_NOAA_CUDEM", 'reproj_method': 'bilinear', 'zmax': 10},
]

# Interpolate the elevation rasters to the grid
dep = mod.setup_dep(datasets_dep=datasets_dep)
mod.write_grid()
print('Done interpolating topography to the grid')
_ = mod.plot_basemap(fn_out='terrain.png', variable="dep", bmap="sat", zoomlevel=5)
plt.close()

# Setup Mask
mod.setup_mask_active(mask=mask_dataset, zmin=-30, reset_mask=True)
print('Done with setting up mask')

# Identify boundary cells
mod.setup_mask_bounds(btype='waterlevel',
                      include_mask='carolinas_coastal_wb',
                      connectivity=8,
                      zmin=-50,
                      zmax=0,
                      reset_bounds=True)

mod.setup_mask_bounds(btype='outflow',
                      include_mask='outflow_bc',
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
     'reclass_table': os.path.join(cat_dir, 'lulc/nlcd/nlcd_mapping_mean.csv')}]

# This is only necessary when not using the subgrid
# mod.setup_manning_roughness(datasets_rgh=datasets_rgh)
# _ = mod.plot_basemap(fn_out='mannings.png', variable="manning", plot_bounds=False, bmap="sat", zoomlevel=12)
# plt.close()
# print('Done with setting up mannings roughness')

# Setup Curve Number Infiltration without recovery - A
# mod.setup_cn_infiltration(cn='gcn250',
#                           antecedent_moisture='avg')
# _ = mod.plot_basemap(fn_out='scs_curvenumber.png', variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
# plt.close()
# mod.write()

# Setup Curve Number Infiltration with recovery - B
"""Setup model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS
including recovery term based on the soil saturation

Parameters
---------
lulc : str, Path, or RasterDataset
    Landuse/landcover data set
hsg : str, Path, or RasterDataset
    HSG (Hydrological Similarity Group) in integers
ksat : str, Path, or RasterDataset
    Ksat (saturated hydraulic conductivity) [mm/hr]
reclass_table : str, Path, or RasterDataset
    reclass table to relate landcover with soiltype
effective : float
    estimate of percentage effective soil, e.g. 0.50 for 50%
block_size : float
    maximum block size - use larger values will get more data in memory but can be faster, default=2000
"""

mod.setup_cn_infiltration_with_ks(lulc='nlcd_2016',
                                  hsg='gNATSGO_hsg_conus',
                                  ksat='gNATSGO_ksat_DCP_0to20cm_carolinas',
                                  reclass_table=r'/projects/sfincs/data/soil/surrgo/CN_Table_HSG_NLCD.csv',
                                  effective=0.75,
                                  block_size=2000)
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
        'dthisout': '900',
        'tspinup': '86400',
        'dtmaxout': '99999999',

        # 'advection': '1',
        # 'viscosity': '1',
        'alpha': '0.5',
        'theta': '0.9',
        'huthresh': '0.05',

        'min_lev_hmax': '-20',
        'zsini': '0.25',
        'stopdepth': '100',

        'rhoa': '1.25',
        'cd_nr': '0',
        'cdnrb': '3',
        'cdwnd': '0 28 50',
        'cdval': '0.0010 0.00250 0.0025',

        'obsfile': 'sfincs.obs',

        'twet_threshold': '0.1',
        'storetwet': '1',
        # 'storevel': '1',
        # 'storemeteo': '1',
        # 'storecumprcp': '1',
        'storevelmax': '1',
        # 'storemaxwind': '1',
    }
)
print(mod.config)
mod.write_config(config_fn='sfincs.inp')

# Setup Structures
mod.setup_structures('levees_carolinas',
                     stype='weir',
                     dep='Carolinas_10m_DEM_USGS_NED',
                     buffer=None,
                     merge=False,
                     dz=0.0
                     )
data = mod.geoms['weir']
data.to_file(os.path.join(os.getcwd(), root, 'gis', 'weir.shp'))
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

# Setup discharge forcing
mod.setup_discharge_forcing(geodataset='usgs_discharge_florence',
                            merge=False,
                            buffer=2000)
mod.write_forcing(data_vars='dis')
gdf_locs = mod.forcing['dis'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['dis'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
print('Write dis')

# Setup gridded precipitation forcing
mod.setup_precip_forcing_from_grid(precip='mrms_tc_florence',
                                   aggregate=False)
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

# Setup observation points
# Copy over obs file created already with river and gage locs and names

# Write and plot model
_ = mod.plot_basemap(fn_out="basemap.png", bmap="sat")
plt.close()
mod.write()
print(f'Script run time before subgrid: {datetime.datetime.now() - startTime}')

# Add Setup Model Subgrid
print('Writing subgrid... will take a while!!!')
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
print(f'Done writing model to {mod.root}')
print(f'Script run time total: {datetime.datetime.now() - startTime}')

