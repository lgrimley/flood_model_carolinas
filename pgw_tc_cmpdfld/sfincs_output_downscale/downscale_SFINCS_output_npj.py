import xarray as xr
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import hydromt
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
mpl.use('TkAgg')
plt.ion()
import time
from scipy import ndimage
import pandas as pd
import numpy as np
import dask.array

start_time = time.time()

# code upgrades that need to happen
# combine/consolidate the flood area calculation and getting the flood depths

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


wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final'
os.chdir(wdir)

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml, yml_base])
# Read in the data catalog to get the model and basin geom
basins = cat.get_geodataframe(data_like=r'.\downscale_test\masks\huc6_basins.shp')
basins = basins.to_crs(epsg=32617)

# Data specifics
chunks_size = {'x': 5000, 'y': 5000}
res = 200
hmin = 0.05

'''' Create basin and water body masks if they don't exist '''

water_mask = rf'.\downscale_test\masks\water_mask_sbgRes{res}m.tif'
if os.path.exists(water_mask) is False:
    dep = cat.get_rasterdataset(rf'..\..\..\sfincs\subgrid\dep_subgrid_{res}m.tif')

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
basin_mask = rf'.\downscale_test\masks\basin_mask_sbgRes{res}m.tif'
if os.path.exists(basin_mask) is False:
    dep = cat.get_rasterdataset(rf'..\..\..\sfincs\subgrid\dep_subgrid_{res}m.tif')
    mask = dep.raster.rasterize(basins, 'index', nodata=-128, all_touched=False)
    mask.rio.write_crs('EPSG:32617', inplace=True)
    mask.raster.to_raster(basin_mask, nodata=-128, dtype='int8')

calc_fld_extent = False
if calc_fld_extent is True:
    for clim in ['present', 'future']:
        for storm in ['flor','matt','floy']:
            start_time = time.time()
            mdf = pd.DataFrame()
            for i in range(len(basins.index)):
                clip_geom = basins[basins.index == i]

                wb_mask = cat.get_rasterdataset(water_mask, crs=32617, chunks=chunks_size, geom=basins)
                basin_mask = cat.get_rasterdataset(basin_mask, crs=32617, chunks=chunks_size, geom=clip_geom)

                if clim == 'future':
                    # Load in the full data
                    zsmax_ds = cat.get_rasterdataset(os.path.join(wdir,'ensemble_mean','fut_ensemble_zsmax_mean.nc'),
                                                     geom=clip_geom, chunks=chunks_size)
                    attr_ds = cat.get_rasterdataset(os.path.join(wdir,'ensemble_mean','processes_classified_ensmean_mean.nc'),
                                                    crs=32617, geom=clip_geom, chunks=chunks_size)
                    # Selected run, mask out data beyond the shapefile
                    zsmax_da = zsmax_ds.sel(run=f'{storm}_fut_compound_mean')
                    attr_da = attr_ds.sel(run=f'{storm}_fut_ensmean')
                else:
                    # Load in the full data
                    zsmax_ds = cat.get_rasterdataset('pgw_zsmax.nc', geom=clip_geom, chunks=chunks_size)
                    attr_ds = cat.get_rasterdataset(os.path.join(wdir,'process_attribution','processes_classified.nc'),
                                                    crs=32617, geom=clip_geom, chunks=chunks_size)
                    # Selected run, mask out data beyond the shapefile
                    zsmax_da = zsmax_ds.sel(run=f'{storm}_pres_compound')
                    attr_da = attr_ds.sel(run=f'{storm}_pres')

                elevation_da = cat.get_rasterdataset(rf'..\..\..\sfincs\subgrid\dep_subgrid_{res}m.tif',
                                                     geom=clip_geom, chunks=chunks_size)
                # Regrid the data
                print('Working to regrid the data...')
                rda_zsmax = resized_gridded_output(da_source=zsmax_da,da_target=elevation_da, output_type='float32')
                rda_attr = resized_gridded_output(da_source=attr_da, da_target=elevation_da, output_type='int8')

                # Mask out the data we don't want to deal with
                rda_zsmax_mask1 = rda_zsmax.where(basin_mask.data == i)
                rda_zsmax_mask2 = rda_zsmax_mask1.where(wb_mask.data != 1)

                # Keep elevations above the ground
                zsmask = (rda_zsmax_mask2.data > elevation_da.data)   # np.nan cells return false
                rda_zsmax = rda_zsmax_mask2.where(zsmask)

                # Load the data array into memory
                rda_attr = rda_attr.where(zsmask).astype(dtype='int8').compute()

                data = rda_attr
                expected_values = [1, 2, 3, 4] # Define unique integer values you expect
                counts = {}
                for val in expected_values:
                    # Mask where data == val, skipping NaNs
                    mask = (data == val) & ~dask.array.isnan(data)

                    # Convert to integers (True=1, False=0) and sum
                    count = mask.sum().compute()
                    counts[val] = int(count)

                name = clip_geom['Name'].item()
                df = pd.DataFrame(list(counts.items()), columns=['Value', f'{name}'])
                df.set_index('Value', inplace=True, drop=True)

                mdf = pd.concat(objs=[mdf,df], axis=1, ignore_index=False)
                print(f'Done with {name}')

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time} seconds")


            fld_cells = mdf
            fld_area = fld_cells * (res * res) / (1000 ** 2)  # square km
            fld_area = fld_area.round(2)
            fld_area = fld_area.T
            fld_area.columns=['Coastal','Coastal-Compound','Runoff','Runoff-Compound']
            fld_area['Total'] = fld_area.sum(axis=1)
            fld_area['Compound'] = fld_area[['Coastal-Compound','Runoff-Compound']].sum(axis=1)
            fld_area.to_csv(rf'.\downscale_test\{storm}_{clim}_fld_area_{res}m.csv')

calc_fld_depths = True
if calc_fld_depths is True:
    clip_geom = basins
    wb_mask = cat.get_rasterdataset(water_mask, crs=32617, chunks=chunks_size, geom=clip_geom)
    basin_mask = cat.get_rasterdataset(basin_mask, crs=32617, chunks=chunks_size, geom=clip_geom)
    for clim in ['present', 'future']:
        for storm in ['flor', 'floy','matt']:
            if clim == 'future':
                # Load in the full data
                zsmax_ds = cat.get_rasterdataset(os.path.join(wdir, 'ensemble_mean', 'fut_ensemble_zsmax_mean.nc'),
                                                 geom=clip_geom, chunks=chunks_size)
                attr_ds = cat.get_rasterdataset(
                    os.path.join(wdir, 'ensemble_mean', 'processes_classified_ensmean_mean.nc'),
                    crs=32617, geom=clip_geom, chunks=chunks_size)
                # Selected run, mask out data beyond the shapefile
                zsmax_da = zsmax_ds.sel(run=f'{storm}_fut_compound_mean')
                attr_da = attr_ds.sel(run=f'{storm}_fut_ensmean')
            else:
                # Load in the full data
                zsmax_ds = cat.get_rasterdataset('pgw_zsmax.nc', geom=clip_geom, chunks=chunks_size)
                attr_ds = cat.get_rasterdataset(
                    os.path.join(wdir, 'process_attribution', 'processes_classified.nc'),
                    crs=32617, geom=clip_geom, chunks=chunks_size)

                # Selected run, mask out data beyond the shapefile
                zsmax_da = zsmax_ds.sel(run=f'{storm}_pres_compound')
                attr_da = attr_ds.sel(run=f'{storm}_pres')

            elevation_da = cat.get_rasterdataset(rf'..\..\..\sfincs\subgrid\dep_subgrid_{res}m.tif',
                                                 geom=clip_geom, chunks=chunks_size)

            if elevation_da.shape == zsmax_da.shape is True:
                print('Elevation DEM and water level output have the same shape.')
                rda_zsmax = zsmax_da
                rda_attr = attr_da
            else:
                # Regrid the data
                print('Working to regrid the data...')
                rda_zsmax = resized_gridded_output(da_source=zsmax_da, da_target=elevation_da, output_type='float32')
                rda_attr = resized_gridded_output(da_source=attr_da, da_target=elevation_da, output_type='int8')
                rda_zsmax = rda_zsmax.drop_vars('run')
                rda_attr = rda_attr.drop_vars('run')

            elevation_da.name = 'gnd_elevation'
            rda_zsmax.name = 'zsmax'
            rda_attr.name = 'attr'
            wb_mask.name = 'wb_mask'
            basin_mask.name = 'basin_mask'
            #da_list = [elevation_da, wb_mask, basin_mask, rda_zsmax, rda_attr]
            # Put all the data arrays on the same dimensions as the reference
            # (sometimes there are small rounding error differences in x,y)
            #ref = da_list[0]
            #elevation_da, wb_mask, basin_mask, rda_zsmax, rda_attr = [da.assign_coords({dim: ref.coords[dim] for dim in ref.dims}) for da in da_list]
            #ds = xr.Dataset({da.name: da for da in aligned_list})
            #ds = ds.rio.write_crs('EPSG:32617', inplace=True)

            print('Masking and calculating...')
            # Mask out the water body grid cells (wb cell == 1)
            mask = (wb_mask != 1)
            zsmax_masked = rda_zsmax.where(mask)
            # Mask out water levels below the ground elevation
            zsmax_masked = zsmax_masked.where(zsmax_masked > elevation_da)
            zsmax_masked.rio.write_crs(32617, inplace=True)

            # Calculate the depth above the ground
            # Mask out depths smaller than the selected threshold
            # Mask out the really large depths -- quarries or from model edge along the coastline
            hmax = (zsmax_masked - elevation_da)
            hmax.rio.write_crs(32617, inplace=True)
            mask = (hmax > hmin) & (hmax <= 10)
            hmax_masked = hmax.where(mask)
            hmax_masked.rio.write_crs(32617, inplace=True)

            # Mask out the attribution code data array
            attr_masked = rda_attr.where(mask).astype(dtype='int8')
            attr_masked.rio.write_crs(32617, inplace=True)

            # Output hmax tif for checking
            name = f'{storm}_{clim}'
            hmax_masked.raster.to_raster(fr'.\downscale_test\depths\{name}_hmax_sbgRes{res}m.tif', nodata=np.nan)

            # Get the depth data for the entire domain and calculate stats
            print('Pulling all depths and calculating percentiles...')
            depths = hmax_masked.values
            depths = pd.DataFrame(depths[~np.isnan(depths)])
            df = depths.describe(percentiles=[0.5, 0.9, 0.95])

            # Now loop through and calculate the depth stats and extent for the 3 flood processes
            stat_ls = [df]
            stat_id = [name]
            expected_values = [1, 2, 3]  # 1 = coastal, 2 and 4 = compound, 3 = runoff
            for val in expected_values:
                # Mask where data == val, skipping NaNs
                if val == 2:
                    mask_attr = (attr_masked == val) | (attr_masked == 4) & ~dask.array.isnan(hmax_masked)
                else:
                    mask_attr = (attr_masked == val) & ~dask.array.isnan(hmax_masked)

                depths = hmax_masked.where(mask_attr).values
                depths = pd.DataFrame(depths[~np.isnan(depths)])
                stats = depths.describe(percentiles=[0.5, 0.9, 0.95])

                stat_id.append(f'{name}_attr{val}')
                stat_ls.append(stats)
                print(f'Done with {name} attr code {val}')

            # Save the depth stats and flood extent for the storm to a csv
            mdf = pd.concat(objs=stat_ls, axis=1, ignore_index=False)
            mdf.columns = stat_id
            mdf = mdf.T
            mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 **2)
            mdf.to_csv(fr'.\downscale_test\depths\{name}_depth_stats_sbgRes{res}m.csv')


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for entire script: {elapsed_time} seconds")

# for i in range(len(basins.index)):
#     basin_name = basins['Name'].loc[i]
#     mask2 = (basin_mask.data == i) & mask & ~dask.array.isnan(hmax)
#     depths = hmax.where(mask2).values
#     depths2 = pd.DataFrame(depths[~np.isnan(depths)])
#     stats = depths2.describe(percentiles=[0.5, 0.9, 0.95])
#     stat_id.append(f'{name}_attr{val}_{basin_name}')
#     stat_ls.append(stats)
#     print(f'Done with {name} attr code {val} for {basin_name}')


wd = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\downscale_test\depths'
dfs = []
for res in [5, 20, 200]:
    os.chdir(os.path.join(wd,f'sbgRes{res}m'))
    files = os.listdir()
    files = [f for f in files if f.endswith(".csv")]
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df['gridRes'] = res
        dfs.append(df)

combined_df = pd.concat(dfs, axis=0, ignore_index=False)


split_index = combined_df.index.str.split('_', expand=True)
combined_df.index = pd.MultiIndex.from_tuples(split_index, names=['storm', 'climate', 'attrCode'])
combined_df = combined_df.round(3)
combined_df.to_csv('downscale_comparsion_depth_fldArea.csv')
