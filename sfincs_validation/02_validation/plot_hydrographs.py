from hydromt_sfincs import SfincsModel
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout


def clean_obs_coords(obs_df, source_crs, target_crs):
    # Clean up the observation data and the coordinates
    if 'geometry' in list(obs_df.coords):
        pts = gpd.GeoDataFrame(obs_df.station.values,
                               geometry=obs_df.geometry.values,
                               crs=source_crs)
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.geometry.values = pts.geometry
    else:
        pts = gpd.GeoDataFrame(obs_df.station,
                               geometry=gpd.points_from_xy(x=obs_df.x.values,
                                                           y=obs_df.y.values,
                                                           crs=source_crs))
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.x.values = pts.geometry.x
        obs_df.y.values = pts.geometry.y

    return pts, obs_df


# Load data catalog and model results
model_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter1_FlorenceValidation\sfincs_models\mod_v4_flor' \
             r'\ENC_200m_sbg5m_avgN_adv1_eff75'
mod = SfincsModel(root=model_root, mode='r')
cat = mod.data_catalog
mod.read_results(fn_his='sfincs_his.nc')
print(mod.results.keys())

# Get the station data for querying SFINCS results
mod_zs_da = mod.results['point_zs']
mod_zs_lookup = pd.DataFrame()
mod_zs_lookup['station_id'] = mod_zs_da['station_id'].values
mod_zs_lookup['station_name'] = [x.decode('utf-8').strip() for x in mod_zs_da['station_name'].values]
mod_zs_lookup['data_source'] = [x.rsplit('_', 1)[0] for x in mod_zs_lookup['station_name']]
mod_zs_lookup['data_source_id'] = [x.split('_')[-1] for x in mod_zs_lookup['station_name']]


# LOAD THE OBSERVED WATER LEVEL TIMESERIES
agency ='USGS'
obs_dataset = r'Z:\Data-Expansion\users\lelise\data\storm_data\hurricanes\2018_florence\waterlevel\carolinas_usgs_waterlevel_20180815_20181015_DATA.nc'
obs_da = cat.get_geodataset(obs_dataset, geom=mod.region, variables=["waterlevel"], crs=4326)

pts, obs = clean_obs_coords(obs_df=obs_da, source_crs=4326, target_crs=mod.crs.to_epsg())

# Loop through the observation locations and extract model data
mod_zs_lookup_sub = mod_zs_lookup[mod_zs_lookup['data_source'] == agency]

ids = mod_zs_lookup_sub['data_source_id'].values.tolist()
bc_ids = ['2130910', '2129000', '2102000', '2098206',
          '2102192', '208773375', '2087500', '2090380', '208250410']
import numpy as np
sel_ids = np.setdiff1d(ids,bc_ids)
group_size = 10
groups = [sel_ids[i:i + group_size] for i in range(0, len(sel_ids), group_size)]
counter = 0
for group in groups:
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8,10), sharex=True, sharey=False)
    axs = axs.flatten()
    for i in range(len(group)):
        data_source_id = group[i]

        # Load the observed and modeled data for the select gage
        obs_zs = obs.sel(station=int(data_source_id))
        # Get the model data
        index = mod_zs_lookup_sub[mod_zs_lookup_sub['data_source_id'] == f'{data_source_id}'].index.item()
        mod_zs = mod_zs_da.sel(stations=index)
        # # Add observed and modeled data into a single dataframe
        obs_df = pd.DataFrame(data=obs_zs.values, index=obs_zs.time.values, columns=['Observed'])
        mod_df = pd.DataFrame(data=mod_zs.values, index=mod_zs.time.values, columns=['Modeled'])
        merged_df = pd.concat([obs_df, mod_df], axis=1)
        merged_df.dropna(inplace=True)
        #ss, _ = calculate_stats(station_id=data_source_id, df=merged_df, tstart=None, tend=None)

        ax = axs[i]
        merged_df['Observed'].plot(ax=ax, color='black', marker='.', linestyle='none', alpha=0.8, label='Observed')
        merged_df['Modeled'].plot(ax=ax, color='red', linestyle='-', label='Modeled')
        ax.set_title(fr'USGS Gage: {data_source_id}', fontsize=9)
        ax.set_ylabel('Water Level (m+NAVD88)', fontsize=9)
        if i==0:
            ax.legend()
        print(data_source_id)

    outfile = rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter1_FlorenceValidation\sfincs_models\mod_v4_flor\ENC_200m_sbg5m_avgN_adv1_eff75\validation\hydrographs\group_bc_{counter}.png'
    plt.tight_layout()
    plt.savefig(outfile, dpi=255)
    plt.close()
    counter += 1






