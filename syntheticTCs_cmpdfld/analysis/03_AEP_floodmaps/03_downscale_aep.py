import xarray as xr
import os
import hydromt
import time
from scipy import ndimage
import pandas as pd
import numpy as np
import dask.array



def compare_da_dimensions(da1, da2):
    m1 = mask.dims == rda_zsmax.dims
    print(f'Dimension names and order match: {m1}')

    m2 = all([da1.coords[dim].equals(da2.coords[dim]) for dim in da1.dims])
    print(f'Each coordinate along every dimension is the same for da1 and da2: {m2}')

    m3 = da1.broadcast_equals(da2)
    print(f'Coords, dims and shape are broadcast-compatible: {m3}')
    if m3 is False:
        da2_reindexed = da2.reindex_like(da1, method='nearest')
        print('Reindexed da2 to match da1')
        return da2_reindexed


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


def get_depth_extent_stats(hmax_da: xr.DataArray, run_id, attr_da: xr.DataArray=None,
                           res: int=None) -> pd.DataFrame:
    # Get the depth data for the entire domain and calculate stats
    print('Pulling all depths and calculating percentiles...')
    depths = hmax_da.values
    depths = pd.DataFrame(depths[~np.isnan(depths)])
    df = depths.describe(percentiles=[0.5, 0.9, 0.95])

    # Now loop through and calculate the depth stats and extent for the 3 flood processes
    stat_ls = [df]
    stat_id = [run_id]
    if attr_da is not None:
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



#wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS'
wdir = r'/projects/sfincs/syntheticTCs_cmpdfld/MODEL_OUTPUTS'
os.chdir(wdir)

# Read in the data catalog to get the model and basin geom
#data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
data_catalog_yml = r'/projects/sfincs/data/data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
# Read in the data catalog to get the model and basin geom
basins = cat.get_geodataframe(data_like=r'./masks/basins_shp/huc6_basins.shp')
basins = basins.to_crs(epsg=32617)

# Data specifics
chunks_size = {'x': 5000, 'y': 5000}
res = 20
hmin = 0.05

'''' Create basin and water body masks if they don't exist '''
#elevation_file = rf'..\03_MODEL_RUNS\subgrid\dep_subgrid_{res}m.tif'
elevation_file = rf'./subgrid/dep_subgrid_{res}m.tif'

water_mask = rf'./masks/water_mask_sbgRes{res}m.tif'
if os.path.exists(water_mask) is False:
    dep = cat.get_rasterdataset(elevation_file)

    # Mask out the cells that are considered water bodies
    coastal_wb = cat.get_geodataframe('carolinas_coastal_wb', geom=basins)
    coastal_wb = coastal_wb.to_crs(32617)
    coastal_wb_clip = coastal_wb.clip(basins)
    coastal_wb_clip['mask'] = 1
    mask1 = dep.raster.rasterize(coastal_wb_clip, "mask", nodata=0, all_touched=True)

    carolinas_nhd_area_rivers = cat.get_geodataframe('carolinas_nhd_area_rivers', geom=basins)
    carolinas_nhd_area_rivers = carolinas_nhd_area_rivers.to_crs(32617)
    carolinas_nhd_area_rivers['mask'] = 1
    mask2 = dep.raster.rasterize(carolinas_nhd_area_rivers, "mask", nodata=0, all_touched=True)

    mask = (mask1 + mask2).compute()
    mask = xr.where(cond=mask > 0, x=1, y=0)
    mask = mask.astype('int8')
    mask.rio.write_crs('EPSG:32617', inplace=True)
    mask.raster.to_raster(water_mask, nodata=0)

# Create a basin mask raster at the subgrid resolution
basin_mask = rf'./masks/basin_mask_sbgRes{res}m.tif'
if os.path.exists(basin_mask) is False:
    dep = cat.get_rasterdataset(elevation_file)
    mask = dep.raster.rasterize(basins, 'index', nodata=-128, all_touched=False)
    mask.rio.write_crs('EPSG:32617', inplace=True)
    mask.raster.to_raster(basin_mask, nodata=-128, dtype='int8')


clip_geom = basins
elevation_da = cat.get_rasterdataset(elevation_file, geom=clip_geom, chunks=chunks_size)
wb_mask = cat.get_rasterdataset(water_mask, chunks=chunks_size, geom=clip_geom)
basin_mask = cat.get_rasterdataset(basin_mask, crs=32617, chunks=chunks_size, geom=clip_geom)
wb_mask.rio.write_crs(32617, inplace=True)
basin_mask.rio.write_crs(32617, inplace=True)

for clim in ['ncep', 'canesm_ssp585']:
    for process in ['runoff', 'coastal']:
        start_time = time.time()
        csv_path = fr'./{clim}/aep/{clim}_AEP_floodStats_{process}_sbgRes{res}m_hmin{hmin}m.csv'

        if clim == 'ncep':
            attr_filepath = r'./ncep/aep/ncep_RP_attribution.nc'
            zsmax_filepath = fr'./ncep/aep/ncep_MaxWL_returnPeriods_{process}.nc'
        else:
            attr_filepath = r'./canesm_ssp585/aep/gcm_RP_attribution.nc'
            zsmax_filepath = rf'./canesm_ssp585/aep/projected_MaxWL_returnPeriods_{process}.nc'

        zsmax_ds = cat.get_rasterdataset(zsmax_filepath, crs=32617, geom=clip_geom, chunks=chunks_size)
        attr_ds = cat.get_rasterdataset(attr_filepath, crs=32617, geom=clip_geom, chunks=chunks_size)

        elevation_da.name = 'gnd_elevation'
        wb_mask.name = 'wb_mask'
        basin_mask.name = 'basin_mask'

        counter = 0
        mdf_stats = pd.DataFrame()
        rps = zsmax_ds['return_period'].values
        for rp in [100]: #[10, 25, 50, 100, 250]:
            # Selected run, mask out data beyond the shapefile
            zsmax_da = zsmax_ds.sel(return_period=rp)
            attr_da = attr_ds.sel(return_period=rp)['zsmax_attr']

            zsmax_da.name = 'zsmax'
            attr_da.name = 'attr'
            if clim == 'ncep':
                zsmax_da = zsmax_da.drop_vars(['probability', 'return_period'])
                attr_da = attr_da.drop_vars(['probability', 'return_period','scenario'])
            else:
                zsmax_da = zsmax_da.drop_vars(['rank', 'return_period'])
                attr_da = attr_da.drop_vars(['rank', 'return_period','scenario'])

            zsmax_da.rio.write_crs(32617, inplace=True)
            attr_da.rio.write_crs(32617, inplace=True)

            if elevation_da.shape == zsmax_da.shape is True:
                print('Elevation DEM and water level output have the same shape.')
                rda_zsmax = zsmax_da
                rda_attr = attr_da
            else:
                # Regrid the data
                print('Working to regrid the data...')
                rda_zsmax = resized_gridded_output(da_source=zsmax_da, da_target=elevation_da, output_type='float32')
                if process == 'compound':
                    rda_attr = resized_gridded_output(da_source=attr_da, da_target=elevation_da, output_type='int8')
                else:
                    attr_masked = None

            print('Masking and calculating...')
            # Mask out the water body grid cells (wb cell == 1)
            mask = (wb_mask.data != 1)
            zsmax_masked = rda_zsmax.where(mask)
            # Mask out water levels below the ground elevation
            mask = (zsmax_masked.data > elevation_da.data)
            zsmax_masked = zsmax_masked.where(mask)
            zsmax_masked.rio.write_crs(32617, inplace=True)

            # Calculate the depth above the ground
            # Mask out depths smaller than the selected threshold
            # Mask out the really large depths -- quarries or from model edge along the coastline
            hmax = (zsmax_masked - elevation_da.data)
            hmax.rio.write_crs(32617, inplace=True)
            mask = (hmax > hmin) & (hmax <= 10)
            hmax_masked = hmax.where(mask)
            hmax_masked.rio.write_crs(32617, inplace=True)

            if process == 'compound':
                # Mask out the attribution code data array
                attr_masked = rda_attr.where(mask).astype(dtype='int8')
                attr_masked.rio.write_crs(32617, inplace=True)

            # print('Writing downscaled flood depths to raster...')
            if rp == 100:
                hmax_masked.raster.to_raster(fr'./{clim}/aep/{clim}_RP{rp}_{process}_hmax_sbgRes{res}m.tif', nodata=np.nan)

            mdf1 = get_depth_extent_stats(hmax_da=hmax_masked, attr_da=attr_masked, run_id=f'RP{rp}_Domain', res=res)
            mdf_stats = pd.concat(objs=[mdf_stats, mdf1], axis=0, ignore_index=False)
            for i in range(len(basins.index)):
                basin_name = basins['Name'].loc[i].replace(" ","")
                mask = (basin_mask.data == i)
                hmax_basin = hmax_masked.where(mask)
                run_id = f'RP{rp}_{basin_name}'
                mdf = get_depth_extent_stats(hmax_da=hmax_basin, attr_da=attr_masked, run_id=run_id, res=res)
                mdf_stats = pd.concat(objs=[mdf_stats, mdf], axis=0, ignore_index=False)

            print(f'Done with RP{rp} ({counter} out of {len(rps)})')
            counter += 1

            if os.path.exists(csv_path):
                mdf_stats.to_csv(csv_path, mode='a', header=False, sep=',', index=True)
            else:
                mdf_stats.to_csv(csv_path, mode='w', header=True, sep=',', index=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")


