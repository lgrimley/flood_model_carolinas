import os
from WRF_utils import *


# Load the data catalog
cat = hydromt.DataCatalog(r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml')
state_boundaries = cat.get_geodataframe(
    r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
state_boundaries.to_crs(epsg=4326, inplace=True)
state_boundaries.set_index(keys='NAME', inplace=True)
aoi_model = state_boundaries[state_boundaries.index.isin(['South Carolina', 'North Carolina'])]

'''

Read WRF storm data and store in a dictionary to analyze:
    Reading in netcdfs; there is one per storm, climate, run
    The u-v components are used to calculate the wind speed. 
    The grid is interpolated onto a regular grid so it is easier to clip and mask.

Option to clip the data and mask the data before analysis.

'''

wd = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\hydromt_sfincs_input\wrf_forcing\met'
apply_mask = False #state_boundaries
clip_w_bbox = aoi_model.total_bounds

# Really shouldn't need to change anything below this
os.chdir(wd)
storms = ['floyd', 'matthew', 'florence']
climates = ['present', 'future']
# Create an empty dictionary to store data
runs_dict = {'floyd': {'present': {}, 'future': {}},
             'matthew': {'present': {}, 'future': {}},
             'florence': {'present': {}, 'future': {}}
             }

for storm in storms:
    for climate in climates:
        met_dir = f'{climate}_{storm}'
        for file in os.listdir(os.path.join(os.getcwd(), met_dir)):
            if file.endswith('.nc'):
                run_name = file.split('.')[0].split('_')[-1]
                run_filepath = os.path.join(os.getcwd(), met_dir, file)

                # Read in netcdf of WRF output, calculate wind speed
                # Add wind speed back to the data array
                da = cat.get_rasterdataset(run_filepath)
                da = calc_windspd(da)

                # Interpolate data to a regular grid (just needs to be a tiny bit more regular)
                res = 0.035  # degrees resolution for x,y
                minLat = np.round(da['y'].values.min(), decimals=3)
                maxLat = np.round(da['y'].values.max(), decimals=3)
                yy = np.arange(minLat, maxLat, res) # new y

                minLon = np.round(da['x'].values.min(), decimals=3)
                maxLon = np.round(da['x'].values.max(), decimals=3)
                xx = np.arange(minLon, maxLon, res) # new x

                # Interpolate the da to the new grid
                interpolated_da = da.interp(x=xx, y=yy)

                if clip_w_bbox is not False:
                    interpolated_da = interpolated_da.raster.clip_bbox(bbox=clip_w_bbox)
                    print('Clipping the data to bbox.')

                if apply_mask is not False:
                    apply_mask['mask'] = np.nan
                    mask = interpolated_da.raster.geometry_mask(apply_mask, all_touched=False, invert=True)
                    interpolated_da = interpolated_da.where(mask)
                    #test = test['wind_spd'].max(dim='time')
                    # test.raster.to_raster(
                    #     r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\test.tif')
                    print('Applying mask to the data.')

                # Add dataset to dictionary
                runs_dict[storm][climate][f'{run_name}'] = interpolated_da


variables = [
    #'wind_spd',
    'precip'
]
foldername = 'WRFgrid_CarolinasClip_nomask'

for var in variables:
    if var == 'wind_spd':
        upper_threshold = 1000
        lower_thresholds = np.arange(0, 35, step=5)
    elif var == 'precip':
        upper_threshold = 1000
        lower_thresholds = np.arange(0, 45, step=5)

storms = list(runs_dict.keys())
# Output directory
outdir = rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\wrf_analysis\oct_analysis\{foldername}\{var}'
if os.path.exists(outdir) is False:
    os.makedirs(outdir)
os.chdir(outdir)

for lower_threshold in lower_thresholds:
    stat_df = pd.DataFrame()
    subset_data_stored = pd.DataFrame()
    for storm in storms:
        for climate in list(runs_dict[storms[0]].keys()):
            runs = runs_dict[storm][climate].keys()
            for run in runs:
                # Pull the WRF data for the select storm, climate, and ensemble run
                d = runs_dict[storm][climate][run]

                # Subset the data across the entire storm
                ds = subset_data_by_thresholds(data=d[var], min_threshold=lower_threshold, max_threshold=upper_threshold)

                # Add code to mask/bbox the data here

                df = ds.to_dataframe()  # convert the dataset to a dataframe
                df = df.reset_index()
                df.dropna(inplace=True)  # drop any nan
                df = pd.DataFrame(df[var])
                run_id = f'{storm}_{climate}_{run}'

                # Get stat info on the empirical data
                df = df.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
                df = df.T
                df['climate'] = climate
                df['storm'] = storm
                df['run'] = run
                df['run_id'] = run_id
                stat_df = pd.concat(objs=[stat_df, df], axis=0)

    stat_df.set_index(keys='run_id', drop=True, inplace=True)
    stat_df = stat_df.round(decimals=2).sort_index()
    stat_df.to_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_percentiles.csv')



for lower_threshold in lower_thresholds:
    stat_df = pd.read_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_percentiles.csv',
                          index_col=0)
    fractionalChange_df = pd.DataFrame()
    for storm in stat_df['storm'].unique():
        runs = stat_df[stat_df['storm'] == storm]['run'].unique()
        for run in runs:
            stat_cols = ['mean', '1%', '10%', '25%', '50%', '75%', '90%', '99%']

            x1 = stat_df.loc[f'{storm}_present_{run}', stat_cols]  # Present values
            x2 = stat_df.loc[f'{storm}_future_{run}', stat_cols]  # Future values
            sf = (x2 - x1) / x1  # Scale factor (fractional change)
            sf = pd.DataFrame(sf).astype(float).round(decimals=4)
            sf.columns = [f'{storm}_{run}']
            fractionalChange_df = pd.concat(objs=[fractionalChange_df, sf], axis=1, ignore_index=False)

    fractionalChange_df = fractionalChange_df.T
    fractionalChange_df.to_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_fractionalChange.csv')

    scale_factors_df = (fractionalChange_df + 1).round(decimals=2)
    scale_factors_df.to_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_scaleFactors.csv')

