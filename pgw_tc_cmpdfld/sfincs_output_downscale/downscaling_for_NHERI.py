import os
import numpy as np
import hydromt
import xarray as xr
from scipy import ndimage
import time


def calc_diff_in_zsmax_compound_minus_max_individual(da: xr.DataArray) -> xr.DataArray:
    # Outputs a data array of the diff in water level compound minus max. single driver
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(scenario=['runoff', 'coastal']).max('scenario')

    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(scenario='compound') - da_single_max).compute()
    da_diff.name = 'zsmax_diff'
    da_diff.assign_coords(scenario='total')
    da_diff.attrs = da.attrs
    da_diff.assign_attrs(units='m')
    da_diff.assign_attrs(description='diff in waterlevel compound minus max. single driver')

    return da_diff



def classify_zsmax_by_process(da: xr.DataArray, da_diff: xr.DataArray, hmin: float = 0.1) -> xr.DataArray:
    # Outputs a data array with the zsmax attributed to processes (codes 0 to 4)
    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(scenario='coastal').fillna(0) > da.sel(scenario=['runoff']).fillna(0).max('scenario')
    runoff_mask = da.sel(scenario='runoff').fillna(0) > da.sel(scenario=['coastal']).fillna(0).max('scenario')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()

    da_classified.name = 'zsmax_classified'
    #da_classified = da_classified.assign_coords(scenario='total')
    # da_classified = da_classified.assign_attrs(hmin=hmin,
    #                                            no_class=0, coast_class=1, coast_compound_class=2,
    #                                            runoff_class=3, runoff_compound_class=4)
    # da_classified = da_classified.astype(int)
    return da_classified



def resized_gridded_output(ds: xr.Dataset, elevation_da: xr.DataArray) -> xr.DataArray:
    start_time = time.time()
    target_shape = elevation_da.shape
    var = list(ds.data_vars.keys())[0]
    scenarios = ds.scenario.values
    rdas = []
    for scen in scenarios:
        da = ds.sel(scenario=scen)[var]
        scaling_factors = [target_shape[i] / da.shape[i] for i in range(len(da.shape))]
        ra = ndimage.zoom(input=da, zoom=scaling_factors, order=1,
                                     output='float32', mode='grid-constant',
                                     cval=np.nan, prefilter=False, grid_mode=True)
        rda = xr.DataArray(ra,
                           dims=da.dims,
                           coords={dim: np.linspace(da.coords[dim].min(), da.coords[dim].max(),
                                                       target_shape[i]) for i, dim in enumerate(da.dims)},
                           attrs=da.attrs)
        rda['spatial_ref'] = da['spatial_ref']
        mask = rda.data > sbg.data
        rda2 = rda.where(mask)
        rdas.append(rda2)

    da = xr.concat(objs=rdas, dim='scenario')
    da['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return da



yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=yml_base)
sbg = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')
zsmax = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\pgw_zsmax.nc')
runs = zsmax.run.values


ds = zsmax.sel(run=['flor_pres_coastal','flor_pres_runoff','flor_pres_compound'])
ds = ds.rename({'run': 'scenario'})
ds['scenario'] = ['coastal','runoff','compound']

rda = resized_gridded_output(ds=ds, elevation_da=sbg)
attrs = {'variable':'maximum water surface elevation',
        'units':'m+NAVD88',
        'description':'SFINCS modeled water levels across the Carolinas at a 20m resolution for Hurricane Florence (2018). We hindcast 3 scenarios: runoff, coastal, and compound (e.g., runoff + coastal) processes.',
        'data_doi':'10.17603/ds2-sf10-w836',
        'data_link':'https://www.designsafe-ci.org/data/browser/projects/PRJ-5757/preview',
        'paper_link':'https://www.nature.com/articles/s44304-024-00046-3',
        'contact':'lauren.grimley@unc.edu',
        'year_created':'2024'
        }
rda.attrs = attrs
rda.to_netcdf(r'Z:\Data-Expansion\users\lelise\data_share\SFINCS_OUTPUT\published_on_NHERI_120324\sfincs wse 20m\SFINCS_FlorenceHindcast_WSE_allScenarios.nc')

ddd = []
ccc = []
for storm in ['Florence', 'Matthew', 'Floyd']:
    rda = xr.open_dataarray(f'SFINCS_{storm}Hindcast_WSE_allScenarios.nc')
    da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da=rda)
    da_class = classify_zsmax_by_process(da=rda, da_diff=da_diff, hmin = 0.05)
    da_class = xr.where((da_class == 2) | (da_class == 4), x=5, y=da_class)
    da_class = da_class.astype(np.int32)
    ddd.append(da_diff)
    ccc.append(da_class)
    print(storm)

da = xr.concat(objs=ddd, dim='storm')
da['storm'] = xr.IndexVariable(dims='storm', data=['Florence', 'Matthew', 'Floyd'])
attrs = {'variable':'Peak Water Level Difference',
        'units':'m',
        'description':'difference between SFINCS peak waterlevel compound minus the maximum across the individual drivers (runoff and coastal). Positive values are an increase in water level due to the combined processes.',
        'data_doi':'10.17603/ds2-sf10-w836',
        'data_link':'https://www.designsafe-ci.org/data/browser/projects/PRJ-5757/preview',
        'paper_link':'https://www.nature.com/articles/s44304-024-00046-3',
        'contact':'lauren.grimley@unc.edu',
        'year_created':'2024'
        }
da.attrs = attrs
encoding = {
    'zlib': True,  # Enable compression using zlib
    'complevel': 5,  # Compression level (0 to 9, higher is better compression)
    'shuffle': True  # Enable shuffling for better compression, especially for large arrays
}
da_renamed = da.rename("zsmax_diff")
da_renamed.to_netcdf('peak_waterlevel_difference_comp_minus_indiv.nc', encoding={'zsmax_diff': encoding})


t = xr.open_dataarray(r'Z:\Data-Expansion\users\lelise\data_share\SFINCS_OUTPUT\published_on_NHERI_120324\sfincs wse 20m\SFINCS_FlorenceHindcast_WSE_allScenarios.nc')

