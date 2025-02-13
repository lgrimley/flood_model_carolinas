import sys
import os
import pandas as pd
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')
import hydromt
import xarray as xr

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

''' Processes TCR rainfall by basin '''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis')
ds = xr.open_dataset(r'.\rain\tc_precipitation_stats.nc').compute()
mask = xr.open_dataarray(r'.\rain\basin_mask_TCR.tif')
selected_tcs = ds['tc_id']

# Loop through the basins and calculate the basin average
df_list = []
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i].replace(" ","")

    # Calculate basin cumulative precip
    df = ds['total_precip'].where(mask == i).sum(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    basin_area = mod_domain['area_sqkm'][i]
    df = df * 1e-6 * basin_area
    df.columns = [f'{basin}_CumPrecipKM3']
    df_list.append(df)

    # Calculate average total precipitation in a cell
    df = ds['total_precip'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    df.columns = [f'{basin}_MeanTotPrecipMM']
    df_list.append(df)

    # Calculate max total precipitation in a cell
    df = ds['total_precip'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    df.columns = [f'{basin}_MaxTotPrecipMM']
    df_list.append(df)

    # Calculate max rain rate
    df = ds['max_RR'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    df.columns = [f'{basin}_maxRR']
    df_list.append(df)

    # Calculate the mean rain rate, no threshold
    df = ds['mean_RR'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    df.columns = [f'{basin}_meanRR']
    df_list.append(df)

    # Calculate the mean rain rate using a min threshold of 5 mm/hr
    df = ds['mean_RR_thresh'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns='spatial_ref', inplace=True)
    df.columns = [f'{basin}_meanRRthresh']
    df_list.append(df)

    print(f'Done processing {basin}')

basin_precip_stats = pd.concat(objs=df_list, axis=1, ignore_index=False)
basin_precip_stats = basin_precip_stats.droplevel(1)

basin = 'Domain'
df_list = []
# Calculate basin cumulative precip
df = ds['total_precip'].sum(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
basin_area = mod_domain['area_sqkm'][i]
df = df * 1e-6 * basin_area
df.columns = [f'{basin}_CumPrecipKM3']
df_list.append(df)

# Calculate average total precipitation in a cell
df = ds['total_precip'].mean(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
df.columns = [f'{basin}_MeanTotPrecipMM']
df_list.append(df)

# Calculate max total precipitation in a cell
df = ds['total_precip'].max(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
df.columns = [f'{basin}_MaxTotPrecipMM']
df_list.append(df)

# Calculate max rain rate
df = ds['max_RR'].max(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
df.columns = [f'{basin}_maxRR']
df_list.append(df)

# Calculate the mean rain rate, no threshold
df = ds['mean_RR'].mean(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
df.columns = [f'{basin}_meanRR']
df_list.append(df)

# Calculate the mean rain rate using a min threshold of 5 mm/hr
df = ds['mean_RR_thresh'].mean(dim=['x', 'y']).to_dataframe()
df.drop(columns='spatial_ref', inplace=True)
df.columns = [f'{basin}_meanRRthresh']
df_list.append(df)

rain_stats = pd.concat(objs=df_list, axis=1, ignore_index=False)
print(f'Done processing {basin}')

final = pd.concat(objs=[basin_precip_stats, rain_stats], axis=1, ignore_index=False).round(2)
final.to_csv(r'.\rain\basin_tc_precipitation_stats.csv')


''' Processes Wind by basin '''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis')
ds = xr.open_dataset(r'.\wind\tc_windspeed_stats.nc').compute()
mask = xr.open_dataarray(r'.\wind\basin_mask_wind.tif')
selected_tcs = ds['tc_id']

# Loop through the basins and calculate the basin average
df_list = []
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i].replace(" ","")
    # max windspeed across the basin
    df = ds['max_windspd'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.columns = [f'{basin}_maxWS']
    df_list.append(df)
    print('1')

    # take the mean - max wind speeds
    df = ds['max_windspd'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.columns = [f'{basin}_meanMaxWS']
    df_list.append(df)
    print('2')

    # mean windspeed
    df = ds['mean_windspd'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.columns = [f'{basin}_meanWS']
    df_list.append(df)
    print('3')

    # mean windspeed w/ threshold
    df = ds['mean_windspd_threshold'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.columns = [f'{basin}_meanWSthresh']
    df_list.append(df)
    print('4')

    # mean wind direction
    # Wind conventions: The direction represents the origin of the wind.
    # 0°: Wind coming from the north (i.e., blowing southward).
    # 90°: Wind coming from the east (i.e., blowing westward).
    # 180°: Wind coming from the south (i.e., blowing northward).
    # 270°: Wind coming from the west (i.e., blowing eastward).
    df = ds['mean_direction_deg'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.columns = [f'{basin}_meanDirection']
    df_list.append(df)
    print('5')
    print(f'Done processing {basin}')

basin_windspd_stats = pd.concat(objs=df_list, axis=1, ignore_index=False)
basin_windspd_stats = basin_windspd_stats.droplevel(1)

# Domain wide
df_list = []
basin='Domain'
# max windspeed across the basin
df = ds['max_windspd'].max(dim=['x', 'y']).to_dataframe()
df.columns = [f'{basin}_maxWS']
df_list.append(df)

# take the mean - max wind speeds
df = ds['max_windspd'].mean(dim=['x', 'y']).to_dataframe()
df.columns = [f'{basin}_meanMaxWS']
df_list.append(df)

# mean windspeed
df = ds['mean_windspd'].mean(dim=['x', 'y']).to_dataframe()
df.columns = [f'{basin}_meanWS']
df_list.append(df)

# mean windspeed w/ threshold
df = ds['mean_windspd_threshold'].mean(dim=['x', 'y']).to_dataframe()
df.columns = [f'{basin}_meanWSthresh']
df_list.append(df)

# mean wind direction
df = ds['mean_direction_deg'].mean(dim=['x', 'y']).to_dataframe()
df.columns = [f'{basin}_meanDirection']
df_list.append(df)

windspd_stats = pd.concat(objs=df_list, axis=1, ignore_index=False)
print(f'Done processing {basin}')


final = pd.concat(objs=[basin_windspd_stats, windspd_stats], axis=1, ignore_index=False).round(2)
final.to_csv(r'.\wind\basin_tc_windspd_stats.csv')





