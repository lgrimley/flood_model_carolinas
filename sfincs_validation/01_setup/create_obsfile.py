import os
import datetime
import hydromt
import pandas as pd
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import geopandas as gpd

yml = os.path.join(r'Z:\users\lelise\data', 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\arch_hindcast\flor_pres_compound'
mod = SfincsModel(root=root, mode='r', data_libs=[yml])

# USGS locations
ds = cat.get_geodataset('usgs_waterlevel_florence')
df_usgs = pd.DataFrame()
df_usgs['x'] = ds.x.values
df_usgs['y'] = ds.y.values
df_usgs['owner'] = 'USGS'
df_usgs['sta_id'] = ds['index'].values.astype(int)

# NOAA locations
ds = cat.get_dataframe(data_like=r'Z:\users\lelise\data\gages\ncem\contrails_and_noaa_stationdata_meta_ALL_2022-09'
                                 r'-20_Ian.csv', header=0)
ds['OWNER'].replace('NOAA/NOS', 'NOAA', inplace=True)
ds_noaa = ds[ds['OWNER'] == 'NOAA']
df_noaa = pd.DataFrame()
df_noaa['x'] = ds_noaa.LON.values
df_noaa['y'] = ds_noaa.LAT.values
df_noaa['owner'] = 'NOAA'
df_noaa['sta_id'] = ds_noaa['STATION'].astype(int).values

# NCEM
ds_ncem = ds[ds['OWNER'] == 'NCEM']
df_ncem = pd.DataFrame()
df_ncem['x'] = ds_ncem.LON.values
df_ncem['y'] = ds_ncem.LAT.values
df_ncem['owner'] = 'NCEM'
df_ncem['sta_id'] = ds_ncem['STATION'].astype(str).values
sta_ids_ncem = []
for val in df_ncem['sta_id']:
    try:
        sta_ids_ncem.append(int(val))
        print('yay')
    except:
        sta_ids_ncem.append(val)
df_ncem['sta_id'] = sta_ids_ncem

# USGS Rapid deployment
ds_usgs_rd = cat.get_dataframe(data_like=r'Z:\users\lelise\data\gages\observations\Florence'
                                 r'\usgs_rapid_deployment_FilteredInstruments.csv', header=0)
ds_usgs_rd['owner'] = 'USGS_RD'
ds_usgs_rd['sta_id'] = ds_usgs_rd['index'].astype(str).values
df_usgs_rd = ds_usgs_rd.drop(columns='index')

# Combine data
df_combined = pd.concat([df_usgs, df_noaa, df_usgs_rd, df_ncem], axis='index', ignore_index=True)
gdf_combined = gpd.GeoDataFrame(df_combined,
                                geometry=gpd.points_from_xy(x=df_combined['x'], y=df_combined['y'], crs=4326))
gdf_combined_remDups = gdf_combined.drop_duplicates(subset=['sta_id'], keep='first', ignore_index=False)
gdf_combined_remDups.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\carolinas_gage_locations_all.csv')

# Read back in data and clip to the domain
obs_clipped = mod.data_catalog.get_geodataframe(
    data_like=r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\carolinas_gage_locations_all.csv',
    geom=mod.region,
    crs=4326).to_crs(mod.crs)
obs_clipped['sta_id'] = obs_clipped['owner'] + '_' + obs_clipped['sta_id']
obs_clipped = obs_clipped.drop_duplicates(subset='geometry', keep='first')

with open(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\carolinas_gage_locations.obs', mode='w') as obs_file:
    for i in range(len(obs_clipped)):
        x = obs_clipped.geometry.x.values.round(2)[i]
        y = obs_clipped.geometry.y.values.round(2)[i]
        id = obs_clipped['sta_id'].values[i]
        line_entry = f'{x:<011}    {y:<011}    "{id}"\n'
        obs_file.write(line_entry)
obs_file.close()
