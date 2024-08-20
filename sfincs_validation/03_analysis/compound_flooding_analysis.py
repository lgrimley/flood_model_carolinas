import os
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils

"""
SCRIPT:
DESCRIPTION:
This script processes SFINCS model output to attribute peak water levels to the different flood processes or drivers.
Pieces of this code can be reused for a similar analysis.

AUTHOR: Lauren Grimley
CONTACT: lauren.grimley@unc.edu

"""


def classify_zsmax_by_driver(da, compound_key, runoff_key, coastal_key, name_out, hmin):
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(run=[runoff_key, coastal_key]).max('run')
    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(run=compound_key) - da_single_max).compute()
    da_diff.name = 'diff in waterlevel compound minus max. single driver'
    da_diff.attrs.update(unit='m')

    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(run=coastal_key).fillna(0) > da.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da.sel(run=runoff_key).fillna(0) > da.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_classified.name = name_out

    # Calculate the number of cells that are attributed to the different drivers
    unique_codes, fld_area_by_driver = np.unique(da_classified.data, return_counts=True)

    # Return compound only locations
    da_compound = xr.where(compound_mask, x=1, y=0)
    da_compound.name = name_out

    return da_classified, fld_area_by_driver, da_compound, da_diff


def get_depth_stats(da_depth, var, thresholds):
    df_depth = da_depth.to_dataframe()
    df_depth.dropna(axis=0, inplace=True)
    df_depth = pd.DataFrame(df_depth[var])

    df_ss = pd.DataFrame()
    for threshold in thresholds:
        df_depth_sub = df_depth[df_depth[var] > threshold].astype(float).round(3)
        depth_stats = df_depth_sub.describe(percentiles=[0.05, 0.5, 0.95])
        depth_stats.columns = [f'depth_stats_{threshold}m']
        df_ss = pd.concat([df_ss, depth_stats], axis=1, ignore_index=False)

    return df_ss


# Filepath to data catalog yml
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
os.chdir(r'Z:\users\lelise\projects\Carolinas_SFINCS\Chapter1_FlorenceValidation\sfincs_models\mod_v4_flor')
model_root = r'ENC_200m_sbg5m_avgN_adv1_eff75'
mod = SfincsModel(model_root, mode='r',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])

# Load peak flood maps
# The netcdf of peak water levels was created using the downscale_floodmaps.py script
res = 200
da = xr.open_dataarray(os.path.join(os.getcwd(), 'floodmaps', f'{res}m', 'floodmaps.nc'))
print(da.keys())

# Load in HUC6 watershed boundary
huc_boundary = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape'
                             r'\WBDHU6.shp')
huc_boundary.to_crs(32617, inplace=True)
huc_boundary = huc_boundary[["HUC6", "Name", "geometry"]]

# Create working directory
out_dir = os.path.join(os.getcwd(), 'process_attribution', f'{res}m')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

''' Part 1 - Attribute total flood extent to runoff, coastal, and compound processes '''
outfile_ext = 'all'
part1 = False
if part1 is True:
    out = classify_zsmax_by_driver(da=da,
                                   compound_key='ENC_200m_sbg5m_avgN_adv1_eff75',
                                   runoff_key='ENC_200m_sbg5m_avgN_adv1_eff75_runoff',
                                   coastal_key='ENC_200m_sbg5m_avgN_adv1_eff75_coastal',
                                   name_out=out_dir,
                                   hmin=0.05
                                   )
    da_c, fld_cells_by_driver, da_compound, da_diff = out

    # Write to files
    da_c.to_netcdf(f'flor_peakWL_attributed_{outfile_ext}.nc')
    da_compound.to_netcdf(f'flor_peakWL_compound_extent_{outfile_ext}.nc')
    da_diff.to_netcdf(f'flor_peakWL_compound_minus_maxIndiv_{outfile_ext}.nc')
    da_diff.raster.to_raster(f'flor_peakWL_compound_minus_maxIndiv_{outfile_ext}.tif', nodata=np.nan)

    # Calculate flood area
    fld_area = fld_cells_by_driver.copy()
    res = da_diff.raster.res[0]  # meters
    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = pd.DataFrame(fld_area.T)
    fld_area.index = ['None', 'Coastal', 'Coastal Compound', 'Runoff', 'Runoff Compound']
    fld_area.to_csv(f'flor_peakWL_attributed_area_{outfile_ext}.csv')

    var = 'diff in waterlevel compound minus max. single driver'
    depth_stats = get_depth_stats(da_depth=da_diff, var=var, thresholds=[-9999.0, 0.0, 0.05])
    depth_stats.to_csv(f'flor_peakWL_compound_minus_maxIndiv_stats_{outfile_ext}.csv')

''' Part 2 - Attribute OVERLAND flood extent to runoff, coastal, and compound processes '''
coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb.to_crs(epsg=32617, inplace=True)
coastal_wb = coastal_wb.clip(mod.region)
coastal_wb["water"] = 0
coastal_wb = da.raster.rasterize(coastal_wb, "water", nodata=1, all_touched=False)

# Remove water bodies
da = da.where(coastal_wb == 1)
outfile_ext = 'land'
part2 = True
if part2 is True:
    out = classify_zsmax_by_driver(da=da,
                                   compound_key='ENC_200m_sbg5m_avgN_adv1_eff75',
                                   runoff_key='ENC_200m_sbg5m_avgN_adv1_eff75_runoff',
                                   coastal_key='ENC_200m_sbg5m_avgN_adv1_eff75_coastal',
                                   name_out=out_dir,
                                   hmin=0.05
                                   )
    da_c, fld_cells_by_driver, da_compound, da_diff = out

    # Write to files
    da_c.to_netcdf(f'flor_peakWL_attributed_{outfile_ext}.nc')
    da_compound.to_netcdf(f'flor_peakWL_compound_extent_{outfile_ext}.nc')
    da_diff.to_netcdf(f'flor_peakWL_compound_minus_maxIndiv_{outfile_ext}.nc')
    da_diff.raster.to_raster(f'flor_peakWL_compound_minus_maxIndiv_{outfile_ext}.tif', nodata=np.nan)

    # Calculate flood area
    fld_area = fld_cells_by_driver.copy()
    res = da_diff.raster.res[0]  # meters
    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = pd.DataFrame(fld_area.T)
    fld_area.index = ['None', 'Coastal', 'Coastal Compound', 'Runoff', 'Runoff Compound']
    fld_area.to_csv(f'flor_peakWL_attributed_area_{outfile_ext}.csv')

    var = 'diff in waterlevel compound minus max. single driver'
    depth_stats = get_depth_stats(da_depth=da_diff, var=var, thresholds=[-9999.0, 0.0, 0.05])
    depth_stats.to_csv(f'flor_peakWL_compound_minus_maxIndiv_stats_{outfile_ext}.csv')

''' Get flood area/depth information by HUC6 basin '''
res = 20  # meters
mdf = pd.DataFrame()
for basin in ['Pamlico', 'Neuse', 'Cape Fear', 'Onslow Bay', 'Lower Pee Dee']:
    sub = huc_boundary[huc_boundary['Name'] == basin]
    sub["basin"] = 1
    b = da.raster.rasterize(sub, "basin", nodata=-9999.0, all_touched=False)
    da2_c = da_c.where(b == 1)
    da2_diff = da_diff.where(b == 1)

    unique_codes, fld_cells_by_driver = np.unique(da2_c.data, return_counts=True)
    fld_area = fld_cells_by_driver.copy()

    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = pd.DataFrame(fld_area)
    fld_area.index = ['None', 'Coastal', 'Coastal Compound', 'Runoff', 'Runoff Compound', 'Compound']
    fld_area.columns = [basin]
    mdf = pd.concat([mdf, fld_area], axis=1, ignore_index=False)

mdf.to_csv(f'flor_peakWL_attributed_area_by_HUC.csv')

''' Part 3 - Attribute OVERLAND flood extent to flood drivers (e.g., forcings) '''


def compute_waterlevel_difference(da, scen_base, scen_keys=None):
    # Computer the difference in water level for compound compared to maximum single driver
    da_single_max = da.sel(run=scen_keys).max('run')
    da1 = (da.sel(run=scen_base) - da_single_max).compute()
    da1.name = 'diff. in waterlevel\ncompound - max. single driver'
    da1.attrs.update(unit='m')
    return da1


da1 = compute_waterlevel_difference(da=da,
                                    scen_base='compound',
                                    scen_keys=['stormTide', 'wind', 'discharge', 'rainfall']
                                    )
dh = 0.05
compound_mask = da1 > dh
surge_mask = da.sel(run='stormTide').fillna(0) > da.sel(
    run=['discharge', 'rainfall', 'wind']).fillna(0).max('run')
coastal_mask = da.sel(run='wind').fillna(0) > da.sel(
    run=['discharge', 'rainfall', 'stormTide']).fillna(0).max('run')
discharge_mask = da.sel(run='discharge').fillna(0) > da.sel(
    run=['wind', 'rainfall', 'stormTide']).fillna(0).max('run')
precip_mask = da.sel(run='rainfall').fillna(0) > da.sel(
    run=['wind', 'discharge', 'stormTide']).fillna(0).max('run')
# precip_mask = np.logical_and(precip_mask, da1 >= 0)

assert ~np.logical_and(precip_mask, surge_mask).any() and ~np.logical_and(precip_mask,
                                                                          coastal_mask).any() and ~np.logical_and(
    precip_mask, discharge_mask).any()

assert ~np.logical_and(discharge_mask, surge_mask).any() and ~np.logical_and(discharge_mask,
                                                                             coastal_mask).any()
assert ~np.logical_and(surge_mask, coastal_mask).any()

da_c = (
        + xr.where(surge_mask, x=compound_mask + 1, y=0)
        + xr.where(coastal_mask, x=compound_mask + 3, y=0)
        + xr.where(discharge_mask, x=compound_mask + 5, y=0)
        + xr.where(precip_mask, x=compound_mask + 7, y=0)
).compute()
da_c.name = None
