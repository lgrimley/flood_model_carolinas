import os
import datetime
import hydromt
import pandas as pd
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import geopandas as gpd

yml = os.path.join(r'Z:\Data-Expansion\users\lelise\data', 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\model_setup')
obs_gage = pd.read_csv('sfincs_gage.obs', delim_whitespace=True, header=None)
obs_gage.columns = ['x', 'y', 'name']

riv_gage = pd.read_csv('river_locs.csv')
riv_gage = gpd.GeoDataFrame(riv_gage,
                            geometry=gpd.points_from_xy(x=riv_gage['xcoord'],y=riv_gage['ycoord'],
                                                        crs=4326)).to_crs(32617)
riv_gage['x'] = riv_gage.geometry.x
riv_gage['y'] = riv_gage.geometry.y
riv_gage['name'] = riv_gage['loc_id']
riv_gage = riv_gage[obs_gage.columns.tolist()]

combined = pd.concat(objs=[obs_gage, riv_gage], axis=0, ignore_index=True)

with open(r'sfincs_obsLocs_gauge_riv.obs', mode='w') as obs_file:
    for i in range(len(combined)):
        x = combined['x'].loc[i].round(2)
        y = combined['y'].loc[i].round(2)
        id = combined['name'].loc[i]
        line_entry = f'{x:<011}    {y:<011}    "{id}"\n'
        obs_file.write(line_entry)
obs_file.close()
