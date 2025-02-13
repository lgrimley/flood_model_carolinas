import os

import numpy as np
import pandas as pd
import xarray as xr

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data')
# Points that were used to extract the ADCIRC reanalysis data
points = pd.read_csv('coastal_locs_EDSReanalysisV2.csv')

# Output from the ADCIRC reanalysis extracting
elements = pd.read_csv('full_data_meta.csv')
data = pd.read_csv('full_data.csv', index_col=0)
time = pd.to_datetime(data.index.values)

# Output netcdf file
fileout = 'EDSReanalysis_V2_1992_2022.nc'

# Metadata
meta = {'description': 'RENCI ADCIRC Reanalysis V2',
        'file_creator': 'Lauren Grimley',
        'contact': 'lauren.grimley@unc.edu',
        'documentation': 'https://renci.github.io/edsreanalysisdoc/',
        'units': 'm',
        'datum': 'msl',
        'category': 'modeled',
        'fill_value': -9999.0,
        'spatial_ref': 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System '
                       '1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System '
                       '1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 '
                       '(G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 ('
                       'G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY['
                       '2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],'
                       'AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
                       'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],'
                       'USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],'
                       'ID["EPSG",4326]]',
        'x_name': 'x',
        'y_name': 'y',
        'geom_format': 'xy',
        'index_dim': 'index',
        'semi_major_axis': 6378137.0,
        'semi_minor_axis': 6356752.314245179,
        'inverse_flattening': 298.257223563,
        'reference_ellipsoid_name': 'WGS 84',
        'longitude_of_prime_meridian': 0.0,
        'prime_meridian_name': 'Greenwich',
        'geographic_crs_name': 'WGS 84',
        'horizontal_datum_name': 'World Geodetic System 1984 ensemble',
        'grid_mapping_name': 'latitude_longitude',
        }

# Create an xarray dataset
ds_out = xr.Dataset(
    data_vars=dict(
        waterlevel=(["time", "index"], data.values),
    ),
    coords=dict(
        x=(['index'], elements['LON'].values),
        y=(['index'], elements['LAT'].values),
        index=(["index"], elements['Point'].values),
        time=time,
        spatial_ref=meta['spatial_ref'],
    ),
    attrs=meta,
)

# Looks like the first day/hour of every year is duplicated when years
# were appended. We drop the first one and keep the second
duplicates = time[time.duplicated()]
ds_out2 = ds_out.drop_duplicates(dim='time', keep='last')

print(ds_out2.index.values)
print(ds_out2.time.min().item())
print(ds_out2.time.max().item())

# Some of the locations having missing data, we interpolate the small gaps
ds_out_interp = ds_out2.interpolate_na(dim='time', method='linear',
                                       use_coordinate=True,
                                       limit=3,
                                       max_gap=pd.Timedelta(value='3h'))

# Interpolate/cleanup the data in case there are gaps
# for g in ds_out.index.values:
#     sta_data = ds_out_interp.sel(index=g).to_dataframe()
#     sta_data = sta_data['waterlevel']
#     missing_points = sta_data.isna().sum()
#     if missing_points > 0:
#         print(f'Total number of missing hourly modeled data at {g} is {missing_points}')
#         mp = sta_data[sta_data.isna()]
#         deltas = mp.index.diff()[1:]
#         print(deltas)

ds_out_interp.to_netcdf(fileout)
print('Done!')



''' 
In the future -- SSP585 2070-2100 
we assume the same tides as the reanalysis so adjust the timestamp
where 1980 == 2070 (add 90 years to the timestamp)
'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\RENCI_EDSReanalysis')
data = xr.open_dataset(r'EDSReanalysis_V2_1992_2022.nc')
df = data['time'].to_dataframe()
df['future'] = df['time'] + pd.DateOffset(years=90)
data['time'] = df['future']
data.to_netcdf(r'EDSReanalysis_V2_1992_2022_with90yrOffset.nc')


