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
mod = SfincsModel(root=root, mode='r+', data_libs=[yml])

# USGS locations
ds = cat.get_geodataset('usgs_waterlevel_florence')
df_usgs = pd.DataFrame()
df_usgs['x'] = ds.x.values
df_usgs['y'] = ds.y.values
df_usgs['owner'] = 'USGS'
df_usgs['index'] = df_usgs['owner'] + '_' + ds['index'].values.astype(str)
df_usgs.drop(columns='owner', inplace=True)

obs = cat.get_geodataset('usgs_waterlevel_florence',
                         geom=mod.region,
                         variables=["waterlevel"],
                         time_tuple=mod.get_model_time()
                         )


# NCEM contrail locations
ds = cat.get_dataframe(data_like=r'Z:\users\lelise\data\gages\ncem\contrails_and_noaa_stationdata_meta_ALL_2022-09'
                                 r'-20_Ian.csv', header=0)
ds['OWNER'].replace('NOAA/NOS', 'NOAA', inplace=True)
df_ncem_noaa = pd.DataFrame()
df_ncem_noaa['x'] = ds.LON.values
df_ncem_noaa['y'] = ds.LAT.values
df_ncem_noaa['index'] = ds['OWNER'].astype(str) + '_' + ds['STATION'].astype(str)

# Bridge locations
# ds = cat.get_geodataframe(
#     data_like=r'Z:\users\lelise\data\NC_State_Agencies\BridgeWatch\Structure_Export_DA_202308.shp')
# ds['type'] = 'BRDG'
# ds['Bridge_ID'] = ds['Bridge_ID'].astype(str)
# ds['Bridge_ID'].fillna(ds['Road'], inplace=True)
# df_nc_brdge = pd.DataFrame()
# df_nc_brdge['x'] = ds.geometry.x.values
# df_nc_brdge['y'] = ds.geometry.y.values
# df_nc_brdge['index'] = ds['type'].astype(str) + '_' + ds['Bridge_ID'].astype(str)
# df_nc_brdge.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\nc_bridge_locs.csv')

# River points
# ds = cat.get_geodataframe(
#     data_like=r'Z:\users\lelise\geospatial\fris\uncRequest09142022\V_E_STREAMCNTRLINE_pts_5km.shp',
#     geom=mod.region).to_crs(4326)
# ds['type'] = 'RVR'
# df_rvr = pd.DataFrame()
# df_rvr['x'] = ds.geometry.x.values
# df_rvr['y'] = ds.geometry.y.values
# df_rvr['index'] = ds['type'].astype(str) + '_' + ds.index.astype(str)
# df_rvr.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\carolinas_rvr_locs.csv')

# USGS Rapid deployment
ds = cat.get_dataframe(data_like=r'Z:\users\lelise\data\gages\observations\Florence'
                                 r'\usgs_rapid_deployment_FilteredInstruments.csv', header=0)
ds['OWNER'] = 'USGS_RD'
ds['index'] = ds['OWNER'].astype(str) + '_' + ds['index'].astype(str)
df_usgs_rd = ds.drop(columns='OWNER')

# Combine data
obs_df_out = pd.concat([df_ncem_noaa], axis='index', ignore_index=True)
obs_df_out.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\carolinas_obs_locs2.csv')

# Read back in data and clip to the domain
clipped = mod.data_catalog.get_geodataframe(
    data_like=r'Z:\users\lelise\projects\ENC_CompFld\carolinas_obs_locs2.csv',
    geom=mod.region,
    crs=4326).to_crs(mod.crs)
clipped.columns = ['name', 'geometry']

c = clipped.drop_duplicates(subset='geometry', keep='first')

with open(r'Z:\users\lelise\projects\ENC_CompFld\sfincs_his_locs2.obs', mode='w') as obs_file:
    for i in range(len(clipped)):
        x = clipped.geometry.x.values.round(2)[i]
        y = clipped.geometry.y.values.round(2)[i]
        id = clipped['name'].values[i]
        line_entry = f'{x:<011}    {y:<011}    "{id}"\n'
        obs_file.write(line_entry)
obs_file.close()
