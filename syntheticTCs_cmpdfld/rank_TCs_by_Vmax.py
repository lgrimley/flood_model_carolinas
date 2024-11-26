import sys

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')

import hydromt
from src.utils import *

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

# Change directory
os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis')

# Load the ADCIRC storm tide gage locations to create a buffer for getting max wind speed
gage_locs = pd.read_csv('.\stormTide\coastline_lores_NCSC.csv', index_col=0)
gdf_loc = gpd.GeoDataFrame(gage_locs, geometry=gpd.points_from_xy(x=gage_locs['x'], y=gage_locs['y'], crs=4326))
gdf_loc_buff = gdf_loc.buffer(distance=1)

# Read in the ADCIRC modeled peaks and get the TC ids across the entire domain
gage_peaks = pd.read_csv('.\stormTide\gage_peaks.csv', index_col=0)
gage_tcs = gage_peaks.index.tolist()

# Get the storm max wind speeds, subset to select TCs if needed
vmax_da = xr.open_dataset(r'.\wind\TC_maxWindSpeeds.nc')
vmax_da_select = vmax_da.sel(run=gage_tcs)
if len(vmax_da_select.run) != len(gage_tcs):
    print('Not all TCs max wind fields found...')

# Get the wind grid, add spatial reference
grid = vmax_da_select.sel(run=gage_tcs[0]).rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(gdf_loc_buff, 'gage_id', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\wind\gageLoc_111kmBuff_mask.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
Vmax = pd.DataFrame(index=gage_tcs)
for i in range(len(gdf_loc_buff.index)):
    id = gdf_loc_buff.index[i]

    # Calculate max wind speed within 200km buffer
    loc_maxVmax = vmax_da_select.where(mask == id).max(dim=['x', 'y']).to_dataframe()
    loc_maxVmax.columns = [f'{id}', 'spatial_ref']
    Vmax = pd.concat(objs=[Vmax, loc_maxVmax[f'{id}']], axis=1, ignore_index=False)

    print(f'Done processing {id}')
Vmax = Vmax.round(2)
Vmax.to_csv(r'.\wind\Vmax_by_gage_loc_111kmBuff_AllTcs.csv')

# subset to the storms that are important for each gage (e.g., generates surge)
# remove the storms that generate 0 surge
gage_peaks_clean = gage_peaks.copy()
gage_peaks_clean[gage_peaks_clean <= 0.0] = np.nan
gage_peaks_clean.to_csv('.\stormTide\gage_peaks_ZerosRemoved.csv')

# Get the Vmax for the TCs that generated surge at each gage location
gage_vmax = Vmax[~gage_peaks_clean.isna()]
gage_vmax.to_csv(r'.\wind\Vmax_by_gage_loc_111kmBuff_SelectTCs.csv')


# Rank the TCs by Vmax for each gage
gage_ranked_TCs = pd.DataFrame()
for g in gage_vmax.columns.values:
    ranked_Vmax = gage_vmax[g].sort_values(ascending=False)
    ranked_TCs = pd.DataFrame(ranked_Vmax.dropna().index)
    print(f'gage {g} TCs: {len(ranked_TCs)}')
    gage_ranked_TCs = pd.concat(objs=[gage_ranked_TCs, ranked_TCs], axis=1, ignore_index=True)
gage_ranked_TCs.columns = gage_vmax.columns.values
gage_ranked_TCs.to_csv(path_or_buf='TCs_VmaxRanked_at_each_gage.csv', index=0)

# Pull the top most intense storms at each gage, get their IDs and the frequency of them being in the top
tc_ids, counts = np.unique(gage_ranked_TCs.loc[0:10,:].values, return_counts=True)
top_tcs = pd.DataFrame([tc_ids, counts]).T
top_tcs.columns = ['tc_id', 'count']

# Get domain-wide max wind speed
vmax_da_select = vmax_da.sel(run=top_tcs['tc_id'].tolist())
vmax_domain = vmax_da_select['wind_speed'].max(['x','y']).to_dataframe()

# Get domain-wide max rain rate
rrmax_da = xr.open_dataset(r'.\rain\precip_TC_maxRainRate.nc')
rrmax_da_select = rrmax_da.sel(run=top_tcs['tc_id'].tolist())
rrmax_domain = rrmax_da_select['precip'].max(['x','y']).to_dataframe()

# Avg total precip
tpmax_da = xr.open_dataset(r'.\rain\precip_TC_AvgTotPrecip.nc')
tpmax_da_select = tpmax_da.sel(run=top_tcs['tc_id'].tolist())
tpmax_domain = tpmax_da_select['precip'].max(['x','y']).to_dataframe()

top_tcs.set_index('tc_id', inplace=True, drop=True)
big_storms = pd.concat(objs=[vmax_domain['wind_speed'], rrmax_domain['precip'], tpmax_domain['precip'], top_tcs],
                       axis=1, ignore_index=False)
mean_surge = []
peak_surge = []
for tc_id in big_storms.index.values:
    s = gage_peaks[gage_peaks.index == tc_id].dropna(axis=1)
    mean_surge.append(np.mean(s))
    peak_surge.append(np.max(s))

big_storms['mean_surge'] = mean_surge
big_storms['peak_surge'] = peak_surge
big_storms = big_storms.round(2)
big_storms.to_csv(r'top10TCs_acrossAllgages.csv')
