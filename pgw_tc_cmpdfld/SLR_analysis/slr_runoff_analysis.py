import os.path
import sys
sys.path.append(r'../sfincs_output')
from pgw_utils import *


# Load SFINCS model
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
mod.read()

# Directory of the SFINCS model results
results_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\03_sfincs_models_SLR_runoff'

# Output directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final')

''' Loop through all model runs and save the peak water level output to a single netcdf'''

# Get peak water levels across all model runs
da_zsmax_list = []
event_ids = []

# Loop through all the model dirs to read the results
for dir in os.listdir(results_dir):
    full_path = os.path.join(results_dir, dir)
    if os.path.isdir(full_path):
        print(f"'{dir}' is a directory.")
        try:
            # Read SFINCS model results
            mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))

            '''Get maximum water surface elevation'''
            zsmax = mod.results["zsmax"].max(dim='timemax')
            # Append this data array to a list
            da_zsmax_list.append(zsmax)
            # Keep a list of the event ids
            event_ids.append(dir)
            print(dir)
        except:
            print(f'Something mess up for {dir}')
    else:
        print(f"'{dir}' is not a directory.")


da_zsmax_slr_runoff = xr.concat(da_zsmax_list, dim='run')
da_zsmax_slr_runoff['run'] = xr.IndexVariable('run', event_ids)
da_zsmax_slr_runoff.to_netcdf('pgw_zsmax_slr_runoff.nc')
da_zsmax_slr_runoff = da_zsmax_slr_runoff.to_dataset()

#da_zsmax_slr_runoff = xr.open_dataset(r'pgw_zsmax_slr_runoff.nc')
da_zsmax_all = xr.open_dataset('pgw_zsmax.nc')

# Drop all future runoff runs
# Get the list of runs in the da_zsmax so we can subset
rids = da_zsmax_all.run.values.tolist()
s = [item for item in rids if 'fut' in item and 'runoff' in item]
result = [item for item in rids if item not in s]
da_zsmax_all = da_zsmax_all.sel(run=result)

# Final dataset we will work from
da_zsmax = xr.concat(objs=[da_zsmax_slr_runoff, da_zsmax_all], dim='run')
da_zsmax.to_netcdf('pgw_zsmax_slr.nc')

# NEXT ANALYSIOS
mask_water = False
# Create a directory to save outputs
if mask_water is True:
    out_dir = os.path.join(os.getcwd(), f'process_attribution_mask_slr')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    ''' Apply a water body mask to the data and recalculate '''
    # Load a SFINCS model and get elevation
    yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
    root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
    mod = SfincsModel(root=root, mode='r', data_libs=[yml_base])
    dep = mod.grid['dep']
    coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
    coastal_wb = coastal_wb.to_crs(mod.crs)
    coastal_wb_clip = coastal_wb.clip(mod.region)
    coastal_wb_clip['mask'] = 1.0
    mask1 = dep.raster.rasterize(coastal_wb_clip, "mask", nodata=0.0, all_touched=False)
    carolinas_nhd_area_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers', geom=mod.region)
    carolinas_nhd_area_rivers = carolinas_nhd_area_rivers.to_crs(mod.crs)
    carolinas_nhd_area_rivers['mask'] = 1.0
    mask2 = dep.raster.rasterize(carolinas_nhd_area_rivers, "mask", nodata=0.0, all_touched=True)
    mask = (mask1 + mask2).compute()
    mask = xr.where(cond = mask > 0.0, x= 1.0,y=0.0)
    da_zsmax = da_zsmax.where(mask == 0.0)
else:
    out_dir = os.path.join(os.getcwd(), f'process_attribution_slr')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

# Generate a list of model run names to loop through and get info
keys = []
for storm in ['flor','floy','matt']:
    k = [f'{storm}_pres_compound', f'{storm}_pres_runoff', f'{storm}_pres_coastal']
    keys.append(k)
for storm in ['flor', 'floy', 'matt']:
    nruns = 8
    if storm == 'matt':
        nruns = 7
    for sf in np.arange(1, nruns, 1):
        for slr in np.arange(1, 6, 1):
            n = f'{storm}_fut_SF{sf}_SLR{slr}'
            k = [f'{n}_compound', f'{n}_runoff', f'{n}_coastal']
            keys.append(k)


''' Attribute peak water levels to a Processes for each run '''
fld_df_area = pd.DataFrame()  # dataframe populated with total flooded area
fld_da_compound_extent = []  # populated with data arrays of the compound areas for each run
fld_da_classified = []
fld_da_diff = []
run_ids = []
for k in keys:
    compound_key, runoff_key, coastal_key = k
    run_name = '_'.join(compound_key.split('_')[:-1])
    da_diff, da_classified, da_compound_extent = classify_zsmax_by_process(da_zsmax=da_zsmax,
                                                                           compound_key=compound_key,
                                                                           runoff_key=runoff_key,
                                                                           coastal_key=coastal_key,
                                                                           name_out=run_name,
                                                                           hmin=0.05
                                                                           )
    # Append the data array to the larger list
    fld_da_diff.append(da_diff)
    fld_da_classified.append(da_classified)
    fld_da_compound_extent.append(da_compound_extent)

    # Write compound extent to raster
    da_compound_extent.raster.to_raster(os.path.join(out_dir, f'{da_compound_extent.name}.tif'), nodata=-9999.0)

    ''' Calculate flood extent attributed to each process '''
    fld_area = calculate_flood_area_by_process(da_classified)
    fld_df_area[f'{da_compound_extent.name}'] = fld_area

    # Append the run ID to the larger list, used for saving as a run index
    run_ids.append(f'{da_compound_extent.name}')
    print(da_compound_extent.name)


# Concatenate the data arrays
outfilenames = [f'zsmaxDiff_maxCmpd_minus_maxIndiv_slr.nc',
                f'processes_classified_slr.nc',
                f'pgw_compound_extent_slr.nc']
outputs = [fld_da_diff, fld_da_classified, fld_da_compound_extent]
for i in range(len(outputs)):
    item = xr.concat(outputs[i], dim='run')
    item['run'] = xr.IndexVariable('run', run_ids)
    item.to_netcdf(os.path.join(out_dir, outfilenames[i]))

# No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
fld_df_area.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
fld_df_area.to_csv(os.path.join(out_dir, f'fldArea_by_process_slr.csv'))


# Analysis directory
workdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final'
#da_zsmax = xr.open_dataset(os.path.join(workdir, 'da_zsmax_all.nc'))
rids = da_zsmax_all.run.values.tolist()

''' Calculate SFINCS ensemble mean/max water levels '''
storms = ['flor', 'matt', 'floy']
scenarios = ['coastal', 'runoff', 'compound']
for ntype in ['mean']:
    # Output directory
    out_dir = os.path.join(workdir, f'ensemble_{ntype}_slr')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    # Empty lists for storing
    ensemble_da_list = []
    run_ids = []
    for storm in storms:
        for climate in ['fut']:
            for scenario in scenarios:
                ''' Calculate the ensemble mean/max '''
                selected_ids = [item for item in rids if 'fut' in item and scenario in item and storm in item]
                sel_zmax = da_zsmax.sel(run=selected_ids)
                if ntype == 'mean':
                    da_ensemble = da_zsmax.sel(run=selected_ids).mean('run')
                    name = f'{storm}_{climate}_{scenario}_{ntype}'

                # Append the summarized ensemble data array to a list
                ensemble_da_list.append(da_ensemble)
                # Append the run ID for indexing at the end
                run_ids.append(name)
                print(name)

    da_ensmean = xr.concat(ensemble_da_list, dim='run')
    da_ensmean['run'] = xr.IndexVariable('run', run_ids)
    da_ensmean.to_netcdf(os.path.join(out_dir, f'fut_ensemble_zsmax_{ntype}.nc'))





