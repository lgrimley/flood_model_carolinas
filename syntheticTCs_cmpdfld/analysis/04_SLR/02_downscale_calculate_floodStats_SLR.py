import xarray as xr
import sys
import os
import hydromt
import time
from scipy import ndimage
import pandas as pd
import numpy as np
import dask.array



def resized_gridded_output(da_source: xr.DataArray, da_target: xr.DataArray,
                           output_type: str='float32') -> xr.DataArray:
    start_time = time.time()
    target_shape = da_target.shape
    scaling_factors = [target_shape[i] / da_source.shape[i] for i in range(len(da_source.shape))]
    # add this if you want to include the tc_id dimension + [1.0]

    ra = ndimage.zoom(input=da_source, zoom=scaling_factors, order=1,
                      output=output_type, mode='grid-constant',
                      cval=np.nan, prefilter=False, grid_mode=True)
    rda = xr.DataArray(ra,
                       dims=da_source.dims,
                       coords={dim: np.linspace(da_source.coords[dim].min(), da_source.coords[dim].max(),
                                                target_shape[i]) for i, dim in enumerate(da_source.dims)},
                       attrs=da_source.attrs)
    rda['spatial_ref'] = da_source['spatial_ref']

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return(rda)


def get_depth_extent_stats(hmax_da: xr.DataArray, attr_da: xr.DataArray, run_id, res: int=None) -> pd.DataFrame:
    # Get the depth data for the entire domain and calculate stats
    print('Pulling all depths and calculating percentiles...')
    depths = hmax_da.values
    depths = pd.DataFrame(depths[~np.isnan(depths)])
    df = depths.describe(percentiles=[0.5, 0.9, 0.95])

    # Now loop through and calculate the depth stats and extent for the 3 flood processes
    stat_ls = [df]
    stat_id = [run_id]
    expected_values = [1, 2, 3]  # 1 = coastal, 2 and 4 = compound, 3 = runoff
    for val in expected_values:
        # Mask where data == val, skipping NaNs
        if val == 2:
            mask_attr = (attr_da == val) | (attr_da == 4) & ~dask.array.isnan(hmax_da)
        else:
            mask_attr = (attr_da == val) & ~dask.array.isnan(hmax_da)

        depths = hmax_da.where(mask_attr).values
        depths = pd.DataFrame(depths[~np.isnan(depths)])
        stats = depths.describe(percentiles=[0.5, 0.9, 0.95])

        stat_id.append(f'{run_id}_attr{val}')
        stat_ls.append(stats)
        print(f'Done with {run_id} attr code {val}')

    # Save the depth stats and flood extent for the storm to a csv
    mdf = pd.concat(objs=stat_ls, axis=1, ignore_index=False)
    mdf.columns = stat_id
    mdf = mdf.T
    mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 **2)
    mdf = mdf.round(3)

    return mdf


start_time = time.time()
wdir = r'/projects/sfincs/syntheticTCs_cmpdfld/MODEL_OUTPUTS'
os.chdir(wdir)

zsmax_filepath = fr'{wdir}/slr_runs/zsmax_canesm_slr.nc'
attr_filepath = fr'{wdir}/slr_runs/attribution_canesm_slr_0.05m.nc'
res = 20
hmin = 0.05
proj_crs = 32617

output_filename = f'canesm_slr_sbg{res}m_hmin{hmin}.csv'
output_filepath = fr'{wdir}/slr_runs/{output_filename}'


#Load in data catalog and all the masks
elevation_file = rf'{wdir}/subgrid/dep_subgrid_{res}m.tif'
water_mask = rf'{wdir}/masks/water_mask_sbgRes{res}m.tif'
basin_mask = rf'{wdir}/masks/basin_mask_sbgRes{res}m.tif'

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'/projects/sfincs/data/data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])

# Load the basins geodataframe
basins = cat.get_geodataframe(data_like=rf'{wdir}/masks/basins_shp/huc6_basins.shp')
clip_geom = basins.to_crs(epsg=proj_crs)

# Chunk size specified for dask
chunks_size = {'x': 5000, 'y': 5000}

# Load the elevation dataarray
elevation_da = cat.get_rasterdataset(elevation_file, geom=clip_geom, chunks=chunks_size)
elevation_da.name = 'gnd_elevation'

# Load the water body mask dataarray
wb_mask = cat.get_rasterdataset(water_mask, chunks=chunks_size, geom=clip_geom)
wb_mask.rio.write_crs(proj_crs, inplace=True)
wb_mask.name = 'wb_mask'

# Load the basin mask dataarray
basin_mask = cat.get_rasterdataset(basin_mask, crs=proj_crs, chunks=chunks_size, geom=clip_geom)
basin_mask.rio.write_crs(proj_crs, inplace=True)
basin_mask.name = 'basin_mask'

# Load in the SFINCS output netcdfs
zsmax_ds = cat.get_rasterdataset(zsmax_filepath, crs=proj_crs, geom=clip_geom, chunks=chunks_size)
attr_ds = cat.get_rasterdataset(attr_filepath, crs=proj_crs, geom=clip_geom, chunks=chunks_size)

print('Done loading all the data, now looping through and calculating overland flood depths and extents...')
counter = 0
mdf_stats = pd.DataFrame()
tc_ids = zsmax_ds['tc_id'].values
for tc_id in tc_ids:
    # Selected run, mask out data beyond the shapefile
    zsmax_da = zsmax_ds.sel(tc_id=tc_id, scenario='compound')
    attr_da = attr_ds.sel(tc_id=tc_id)['zsmax_attr']

    # Clean up the data array before regridded
    zsmax_da.name = 'zsmax'
    attr_da.name = 'attr'
    zsmax_da = zsmax_da.drop_vars(['tc_id', 'scenario'])
    attr_da = attr_da.drop_vars(['tc_id', 'scenario'])
    zsmax_da.rio.write_crs(proj_crs, inplace=True)
    attr_da.rio.write_crs(proj_crs, inplace=True)

    if elevation_da.shape == zsmax_da.shape is True:
        print('Elevation DEM and water level output have the same shape.')
        rda_zsmax = zsmax_da
        rda_attr = attr_da
    else:
        # Regrid the data
        print('Working to regrid the data...')
        rda_zsmax = resized_gridded_output(da_source=zsmax_da, da_target=elevation_da, output_type='float32')
        rda_attr = resized_gridded_output(da_source=attr_da, da_target=elevation_da, output_type='int8')

    print('Masking and calculating...')
    # Mask out the water body grid cells (wb cell == 1)
    mask = (wb_mask.data != 1)
    zsmax_masked = rda_zsmax.where(mask)
    # Mask out water levels below the ground elevation
    mask = (zsmax_masked.data > elevation_da.data)
    zsmax_masked = zsmax_masked.where(mask)
    zsmax_masked.rio.write_crs(proj_crs, inplace=True)

    # Calculate the depth above the ground
    # Mask out depths smaller than the selected threshold
    # Mask out the really large depths -- quarries or from model edge along the coastline
    hmax = (zsmax_masked - elevation_da.data)
    hmax.rio.write_crs(proj_crs, inplace=True)
    mask = (hmax > hmin) & (hmax <= 10)
    hmax_masked = hmax.where(mask)
    hmax_masked.rio.write_crs(proj_crs, inplace=True)

    # Mask out the attribution code data array
    attr_masked = rda_attr.where(mask).astype(dtype='int8')
    attr_masked.rio.write_crs(proj_crs, inplace=True)

    mdf = get_depth_extent_stats(hmax_da=hmax_masked, attr_da=attr_masked, run_id=f'{tc_id}_Domain', res=res)
    mdf_stats = pd.concat(objs=[mdf_stats, mdf], axis=0, ignore_index=False)

    for i in range(len(basins.index)):
        basin_name = basins['Name'].loc[i].replace(" ","")
        mask = (basin_mask.data == i)
        hmax_basin = hmax_masked.where(mask)
        run_id = f'{tc_id}_{basin_name}'
        mdf = get_depth_extent_stats(hmax_da=hmax_basin, attr_da=attr_masked, run_id=run_id, res=res)
        mdf_stats = pd.concat(objs=[mdf_stats, mdf], axis=0, ignore_index=False)

    print(f'Done with {tc_id} ({counter} out of {len(tc_ids)})')
    counter += 1

    if os.path.exists(output_filepath):
        mdf_stats.to_csv(output_filepath, mode='a', header=False, sep=',', index=True)
    else:
        mdf_stats.to_csv(output_filepath, mode='w', header=True, sep=',', index=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for {output_filename}: {elapsed_time} seconds")


