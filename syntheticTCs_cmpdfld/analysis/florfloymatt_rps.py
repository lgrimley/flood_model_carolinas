import numpy as np
import sys
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')
import os
import xarray as xr
import pandas as pd
from src.utils import calculate_flooded_area_by_process
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import hydromt

'''

Load in and calculate storm characteristics for Flor, Floy, Matt 

'''

outputdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\historical_storms\present_meteo'
rr_threshold = 5
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\01_sfincs_models_present')
for storm in ['flor', 'floy', 'matt']:
    if storm == 'floy':
        p = xr.open_dataset(fr'.\{storm}_pres_compound\{storm}_pres_precip2d.nc')
        ws = xr.open_dataset(fr'.\{storm}_pres_compound\{storm}_pres_wind2d.nc')
    else:
        p = xr.open_dataset(fr'.\{storm}_pres_compound\{storm}_hindcast_precip2d.nc')
        ws = xr.open_dataset(fr'.\{storm}_pres_compound\{storm}_hindcast_wind2d.nc')

    # Cumulative precipitation
    precip_sum = p.sum(dim='time')
    precip_sum = precip_sum.rename({'Precipitation': 'total_precip'})
    # Max rain rate across the grid
    max_rain_rate = p.max(dim='time')
    max_rain_rate = max_rain_rate.rename({'Precipitation': 'max_RR'})
    # Mean rain rate across the grid
    mean_rain_rate = p.mean(dim='time')
    mean_rain_rate = mean_rain_rate.rename({'Precipitation': 'mean_RR'})
    # Mean rain rate across the grid above threshold
    mask = p > rr_threshold
    mean_rain_rate_thresh = p.where(mask).mean(dim='time')
    mean_rain_rate_thresh = mean_rain_rate_thresh.rename({'Precipitation': 'mean_RR_thresh'})

    # Write data to netcdf
    ds = xr.merge([precip_sum, max_rain_rate, mean_rain_rate, mean_rain_rate_thresh])
    outfile = os.path.join(outputdir, f'{storm}_precipitation_stats.nc')
    #ds.to_netcdf(outfile)
    print(f'Created {outfile}')

    # Calculate the wind speed (magnitude) and direction (angle)
    wind_direction_rad = np.arctan2(ws['eastward_wind'], ws['northward_wind'])  # Direction in radians
    wind_direction_deg = np.degrees(wind_direction_rad)
    normalized_direction = (wind_direction_deg + 360) % 360
    wnd_direction = normalized_direction.mean(dim='time')
    wnd_direction = wnd_direction.to_dataset()
    wnd_direction = wnd_direction.rename({'eastward_wind':'mean_direction_deg'})

    # Calculate wind speed
    ws['wind_speed'] = np.sqrt((ws['eastward_wind']**2) + (ws['northward_wind']**2))

    # Max wind speed across the grid
    max_wndspd = ws['wind_speed'].max(dim='time')
    max_wndspd = max_wndspd.to_dataset()
    max_wndspd = max_wndspd.rename({'wind_speed':'max_windspd'})

    # Mean wind speed across the grid
    mean_wndspd = ws['wind_speed'].mean(dim='time')
    mean_wndspd = mean_wndspd.to_dataset()
    mean_wndspd = mean_wndspd.rename({'wind_speed':'mean_windspd'})

    # Mean wind speed with threshold
    mask = ws['wind_speed'] > 5
    mean_wndspd_thresh = ws['wind_speed'].where(mask).mean(dim='time')
    mean_wndspd_thresh = mean_wndspd_thresh.to_dataset()
    mean_wndspd_thresh = mean_wndspd_thresh.rename({'wind_speed':'mean_windspd_threshold'})

    # Write data to netcdf
    ds = xr.merge([wnd_direction, max_wndspd, mean_wndspd, mean_wndspd_thresh])
    outfile = os.path.join(outputdir, f'{storm}_windspeed_stats.nc')
    #ds.to_netcdf(outfile)
    print(f'Created {outfile}')


'''

Now get domain and basin meteo info Flor, Floy, Matt 

'''

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)
os.chdir(outputdir)
basin_precip_stats = pd.DataFrame()
for storm in ['flor', 'floy', 'matt']:
    ds = xr.open_dataset(f'{storm}_precipitation_stats.nc').compute()
    grid = ds['total_precip']
    basin_mask = grid.raster.rasterize(mod_domain, "index", nodata=-9999, all_touched=True)

    # Loop through the basins and calculate the basin average
    df_list = []
    df_col_names = []
    for i in range(len(mod_domain.index)+1):
        if i == 5:
            mask = (basin_mask > -1)
            basin= 'Domain'
        else:
            mask = (basin_mask == i)
            basin = mod_domain['Name'][i].replace(" ","")

        # Calculate basin cumulative precip
        df = ds['total_precip'].where(mask).sum(dim=['x', 'y']).item()
        if basin == 'Domain':
            basin_area = mod_domain['area_sqkm'].sum()
        else:
            basin_area = mod_domain['area_sqkm'][i]
        df = df * 1e-6 * basin_area
        df_col_names.append(f'{basin}_CumPrecipKM3')
        df_list.append(df)

        # Calculate average total precipitation in a cell
        df = ds['total_precip'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_MeanTotPrecipMM')
        df_list.append(df)

        # Calculate max total precipitation in a cell
        df = ds['total_precip'].where(mask).max(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_MaxTotPrecipMM')
        df_list.append(df)

        # Calculate max rain rate
        df = ds['max_RR'].where(mask).max(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_maxRR')
        df_list.append(df)

        # Calculate the mean rain rate, no threshold
        df = ds['mean_RR'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanRR')
        df_list.append(df)

        # Calculate the mean rain rate using a min threshold of 5 mm/hr
        df = ds['mean_RR_thresh'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanRRthresh')
        df_list.append(df)

        print(f'Done processing {basin}')

    dd = pd.DataFrame(df_list).T
    dd.columns = df_col_names
    basin_precip_stats = pd.concat([basin_precip_stats, dd], axis=0, ignore_index=True)
basin_precip_stats['storm'] = ['flor', 'floy', 'matt']
final_rr = basin_precip_stats.round(2)
#final_rr.to_csv(r'.\basin_tc_precipitation_stats.csv')

basin_precip_stats = pd.DataFrame()
for storm in ['flor', 'floy', 'matt']:
    ds = xr.open_dataset(f'{storm}_windspeed_stats.nc').compute()
    grid = ds['max_windspd']
    basin_mask = grid.raster.rasterize(mod_domain, "index", nodata=-9999, all_touched=True)

    # Loop through the basins and calculate the basin average
    df_list = []
    df_col_names = []
    for i in range(len(mod_domain.index)+1):
        if i == 5:
            mask = (basin_mask > -1)
            basin= 'Domain'
        else:
            mask = (basin_mask == i)
            basin = mod_domain['Name'][i].replace(" ","")


        # Calculate average total precipitation in a cell
        df = ds['max_windspd'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanMaxWS')
        df_list.append(df)

        # Calculate max total precipitation in a cell
        df = ds['max_windspd'].where(mask).max(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_maxWS')
        df_list.append(df)

        # Calculate max rain rate
        df = ds['mean_windspd'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanWS')
        df_list.append(df)

        # Calculate the mean rain rate, no threshold
        df = ds['mean_windspd_threshold'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanWSthresh')
        df_list.append(df)

        # Calculate the mean rain rate using a min threshold of 5 mm/hr
        df = ds['mean_direction_deg'].where(mask).mean(dim=['x', 'y']).item()
        df_col_names.append(f'{basin}_meanDirection')
        df_list.append(df)

        print(f'Done processing {basin}')

    dd = pd.DataFrame(df_list).T
    dd.columns = df_col_names
    basin_precip_stats = pd.concat([basin_precip_stats, dd], axis=0, ignore_index=True)
basin_precip_stats['storm'] = ['flor', 'floy', 'matt']
final_ws = basin_precip_stats.round(2)
#final_ws.to_csv(r'.\basin_tc_windspeed_stats.csv')


'''

This code calculates the flooded area by each process for the entire model domain and each HUC6 watershed

'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\process_attribution_mask')
da = xr.open_dataset('processes_classified.nc')
da = da.rename({'flor_pres': 'class'})

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)
grid = da.sel(run='flor_pres')['class'].rio.write_crs(32617)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )

# Calculate the total area flooded by each process and save to a pandas dataframe
fld_area_df = pd.DataFrame()
for storm in ['flor', 'floy', 'matt']:
    data = da.sel(run=f'{storm}_pres')['class']
    # Calculate the flooded area by each process across the entire domain
    fld_area = calculate_flooded_area_by_process(da=data)
    fld_area['AOI'] = 'Domain'
    fld_area['storm'] = storm
    fld_area_df = pd.concat(objs=[fld_area_df, fld_area], axis=0, ignore_index=True)
    # Loop through the HUC6 basins and calculate the flooded area by each process
    for i in range(len(mod_domain.index)):
        basin_data = data.where(mask == i)
        fld_area = calculate_flooded_area_by_process(da = basin_data)
        fld_area['AOI'] = mod_domain['Name'][i].replace(" ","")
        fld_area['storm'] = storm
        fld_area_df = pd.concat(objs=[fld_area_df, fld_area], axis=0, ignore_index=True)

# Output the results to a csv
fld_area_df.round(3).to_csv(os.path.join(outputdir, 'overland_flooded_area_table.csv'))


'''

Load in Historic synthetic TC RP data and interpolate the return period for Flor, Floy, Matt values

'''

# Combine the dataframessss
final_ws2 = final_ws.set_index('storm', drop=True)
final_rr2 = final_rr.set_index('storm', drop=True)
fdf = pd.concat([final_rr2, final_ws2], axis=1, ignore_index=False)
final_fld = fld_area_df.set_index('storm', drop=True)

# Now load in the syntheticTC return periods for the historic storms
rp_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob\basin_data'
basins=np.unique([x.split('_')[0] for x in final_rr2.columns])
mdf = pd.DataFrame()
for basin in basins:
    # read in the synthetic data
    basin_data = pd.read_csv(os.path.join(rp_dir,f'{basin}_data_rp_historic.csv'), index_col=0)

    # now do some magic to get the data we just processed for flor, floy, matt into the helpful format
    basin_meteo = fdf.filter(like=basin)
    basin_meteo.columns = [x.split('_')[-1] for x in basin_meteo.columns]
    basin_fld = final_fld[final_fld['AOI'] == basin]
    basin_storms = basin_meteo
    basin_storms['FldArea'] = basin_fld['Compound']

    vars = basin_storms.columns
    basin_storms['Basin'] = basin
    for v in vars:
        basin_storms[f'{v}_rp'] = np.nan
        for i in basin_storms.index:
            values = basin_storms.loc[i,v]
            abs_diff = abs(basin_data[v].fillna(0) - values)
            closest_index = abs_diff.idxmin()
            basin_storms.loc[i, f'{v}_rp'] = basin_data.loc[closest_index, f'{v}_rp']
            out = basin_storms.reset_index()

    mdf = pd.concat([mdf, out], axis=0, ignore_index=True)
mdf.round(3).to_csv(os.path.join(outputdir, 'florfloymatt_rp_historic.csv'))






