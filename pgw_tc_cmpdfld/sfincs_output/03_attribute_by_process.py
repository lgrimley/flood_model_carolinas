import sys

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')
from pgw_utils import *


workdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3'
da_zsmax = xr.open_dataset(os.path.join(workdir, 'pgw_zsmax.nc'))
mask_water = True

# Create a directory to save outputs
if mask_water is True:
    out_dir = os.path.join(workdir, f'process_attribution_mask')
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
    out_dir = os.path.join(workdir, f'process_attribution')
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
            k = [f'{n}_compound', f'{storm}_fut_SF{sf}_runoff', f'{n}_coastal']
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
outfilenames = [f'zsmaxDiff_maxCmpd_minus_maxIndiv.nc',
                f'processes_classified.nc',
                f'pgw_compound_extent.nc']
outputs = [fld_da_diff, fld_da_classified, fld_da_compound_extent]
for i in range(len(outputs)):
    item = xr.concat(outputs[i], dim='run')
    item['run'] = xr.IndexVariable('run', run_ids)
    item.to_netcdf(os.path.join(out_dir, outfilenames[i]))

# No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
fld_df_area.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
fld_df_area.to_csv(os.path.join(out_dir, f'fldArea_by_process.csv'))