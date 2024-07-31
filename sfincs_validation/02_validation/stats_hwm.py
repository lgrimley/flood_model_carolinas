import os
import hydromt_sfincs
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime as dt
import rioxarray as rio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import cartopy.crs as ccrs


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    out = LinearSegmentedColormap.from_list(cmap_name, color_list, N)
    return out


def hwm_to_gdf(csv_file_path, agency, quality=None, dst_crs=None):
    df = pd.read_csv(csv_file_path)

    # If the HWM is downloaded from the USGS
    if agency == 'usgs':
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['longitude'], y=df['latitude'], crs=4326))
        gdf['elev_m'] = gdf['elev_ft'] * 0.3048
        if quality:
            gdf = gdf[gdf['hwm_quality_id'] <= quality]
        if dst_crs:
            gdf.to_crs(dst_crs, inplace=True)
        gdf = gdf[gdf['elev_m'].notna()]

    # If the HWM data is from the NCEM
    elif agency == 'ncem':
        df = pd.read_csv(csv_file_path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon_dd'], y=df['lat_dd'], crs=4326))
        gdf['elev_m'] = gdf['elev_ft'] * 0.3048
        if dst_crs:
            gdf.to_crs(dst_crs, inplace=True)
        gdf = gdf[gdf['elev_m'].notna()]
    return gdf


def calc_stats(observed, modeled):
    mae = abs(observed - modeled).values.mean()
    rmse = ((observed - modeled) ** 2).mean() ** 0.5
    bias = (modeled - observed).values.mean()
    return [round(mae, 2), round(rmse, 2), round(bias, 2)]


# Load in model and read results
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model')
model_roots = [
    # 'ENC_200m_sbg5m_avgN_adv1_eff75',
    # 'ENC_200m_sbg5m_avgN_adv1_eff25',
    # 'ENC_200m_sbg5m_avgN_adv1_eff75',
    'ENC_200m_sbg5m_noChannels_avgN',
    # 'ENC_200m_sbg5m_avgN_LPD1m',
    # 'ENC_200m_sbg5m_noChannels_avgN'
]

for model_root in model_roots:
    mod = SfincsModel(root=model_root, mode='r',
                      data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
    cat = mod.data_catalog
    mod.read_results()

    # Create a directory to save data and figures to
    out_dir = os.path.join(mod.root, 'validation', 'hwm')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    '''' Read in HWM data '''
    # Read USGS HWM data
    storm = 'florence'
    hwm_usgs = hwm_to_gdf(csv_file_path=rf'Z:\users\lelise\geospatial\observations\usgs_{storm}_FilteredHWMs.csv',
                          agency='usgs',
                          quality=3,
                          dst_crs=mod.crs.to_epsg())
    hwm_usgs['data_source'] = 'USGS'
    hwm_usgs = hwm_usgs[hwm_usgs['stateName'].isin(['NC', 'SC'])]

    # Read NCEM HWM file and write to CSV
    if os.path.exists(r'Z:\users\lelise\data\NC_State_Agencies\NCEM_HWM\NCEM_hwm_database_Sep2023.csv') is False:
        hwm_ncem = gpd.read_file(r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\HWM_master_share.gdb')
        hwm_ncem_df = pd.DataFrame(hwm_ncem)
        hwm_ncem_df.drop('geometry', inplace=True, axis=1)
        hwm_ncem_df[hwm_ncem_df.isnull()] = np.nan
        hwm_ncem_df[hwm_ncem_df.isna()] = np.nan
        hwm_ncem_df.to_csv(
            r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\NCEM_hwm_database_Sep2023.csv',
            index=False)

    # Read in NCEM HWM data
    hwm_ncem = hwm_to_gdf(csv_file_path=r'Z:\users\lelise\data\NC_State_Agencies\NCEM_HWM'
                                        r'\NCEM_hwm_database_Sep2023.csv', agency='ncem',
                          quality=None, dst_crs=mod.crs.to_epsg())
    print(hwm_ncem['storm_name'].unique())
    print(hwm_ncem['data_source'].unique())

    # Subset by storm of interest
    hwm_ncem_storm = hwm_ncem[hwm_ncem['storm_name'] == 'Hurricane Florence']
    print(hwm_ncem_storm['data_source'].unique())
    print(hwm_ncem_storm['confidence'].unique())

    # Remove data with Poor or lower quality
    quality_category = ['Unknown/Historical', 'VP: > 0.40 ft', 'Poor: +/- 0.40 ft']
    hwm_ncem_storm = hwm_ncem_storm.loc[~hwm_ncem_storm['confidence'].isin(quality_category)]
    hwm_ncem_storm = hwm_ncem_storm.loc[hwm_ncem_storm['data_source'] == 'NCGS']
    hwm_ncem_storm.columns = ['latitude_dd', 'longitude_dd', 'elev_ft', 'eventName',
                              'data_source', 'hwmQualityName', 'geometry', 'elev_m']

    # Read in gage peaks
    gage_stats = pd.read_csv(os.path.join(mod.root, 'validation', 'waterlevel', 'hydrograph_stats_by_gageID.csv'))
    gage_stats = gpd.GeoDataFrame(gage_stats,
                                  geometry=gpd.points_from_xy(x=gage_stats['x'], y=gage_stats['y'], crs=mod.crs))

    gage_stats = gage_stats[['pe', 'mod_peak_wl', 'obs_peak_wl', 'HUC6', 'source', 'geometry']]
    gage_stats.columns = ['error', 'sfincs_m', 'elev_m', 'Name', 'data_source', 'geometry']
    gage_stats['data_source'] = 'gage_' + gage_stats['data_source']
    gage_stats.drop(columns='Name', inplace=True)

    # Combine datasets
    hwm = pd.concat([hwm_usgs, hwm_ncem_storm, gage_stats], axis=0, ignore_index=True)
    hwm = hwm.drop_duplicates(subset='geometry', keep='first')

    ''' Extract modeled water levels at HWMs and Calc Stats '''
    # Extract peak modeled water levels at the HWM points
    xcoords = hwm.geometry.x.to_xarray()
    ycoods = hwm.geometry.y.to_xarray()
    hwm['sfincs_m'] = mod.results['zsmax'].max(dim='timemax').sel(x=xcoords, y=ycoods,
                                                                  method='nearest').values.transpose()
    # Remove the locations outside the model domain
    hwm = hwm[hwm['sfincs_m'].notna()]
    hwm['error'] = hwm['sfincs_m'] - hwm['elev_m']
    mae, rmse, bias = calc_stats(observed=hwm['elev_m'], modeled=hwm['sfincs_m'])

    ''' Calculate HWM stats by HUC6 Watershed '''
    # Assign to HUC6
    huc_boundary = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape'
                                 r'\WBDHU6.shp')
    huc_boundary.to_crs(mod.crs, inplace=True)
    huc_boundary = huc_boundary[["HUC6", "Name", "geometry"]]
    hwm = gpd.tools.sjoin(left_df=hwm, right_df=huc_boundary, how='left')
    hwm.drop(columns='index_right', inplace=True)

    # Assign to State
    states = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
    states = states[states['NAME'].isin(['South Carolina', 'North Carolina'])]
    states.to_crs(epsg=32617, inplace=True)
    states = states[['STUSPS', 'geometry']]
    hwm = gpd.tools.sjoin(left_df=hwm, right_df=states, how='left')
    hwm.drop(columns='index_right', inplace=True)
    hwm['xcoords'] = hwm.geometry.x.to_xarray()
    hwm['ycoords'] = hwm.geometry.y.to_xarray()

    ''' Extract Depth '''
    sbg = cat.get_rasterdataset(os.path.join(mod.root, 'subgrid', 'dep_subgrid.tif'))
    hmax = utils.downscale_floodmap(
        zsmax=mod.results["zsmax"].max(dim='timemax'),
        dep=sbg,
        hmin=0.05,
        gdf_mask=mod.region,
        reproj_method='bilinear'
    )
    hmax.raster.to_raster(os.path.join(os.getcwd(), 'floodmaps', f'noChannels.tif'), nodata=np.nan)

    xx = hwm['geometry'].x.to_xarray()
    yy = hwm['geometry'].y.to_xarray()
    hwm['sfincs_hmax_m'] = hmax.sel(x=xx, y=yy, method='nearest').values.transpose()
    hwm['sfincs_hmax_m'].fillna(0, inplace=True)
    hwm['height_above_gnd_m'] = hwm['height_above_gnd'] * 0.3048
    hwm['depth_error'] = hwm['sfincs_hmax_m'] - hwm['height_above_gnd_m']

    # Write out all the HWM data
    hwm.to_csv(os.path.join(out_dir, 'hwm_error_all.csv'), index=False)

    ''' Calculate stats by grouping '''
    hwm['domain'] = 'domain'
    stats_by_group = pd.DataFrame()
    for group in ['STUSPS', 'Name', 'domain']:
        for z in hwm[group].unique():
            subset = hwm[hwm[group] == z]
            ss = calc_stats(observed=subset['elev_m'], modeled=subset['sfincs_m'])
            ss.append(z)
            ss = pd.DataFrame(ss).T
            ss.columns = ['mae', 'rmse', 'bias', 'group']
            stats_by_group = pd.concat([stats_by_group, ss], axis=0, ignore_index=True)
    stats_by_group.dropna(axis=0, inplace=True)
    stats_by_group.set_index('group', inplace=True, drop=True)
    stats_by_group.to_csv(os.path.join(out_dir, 'peak_error_stats_by_group.csv'), index=True)

    # Calculate depth stats
    x = len(hwm[~hwm['height_above_gnd_m'].isna()])
    print(f'Depth stats calculated at {x} locations')
    stats_by_group = pd.DataFrame()
    for group in ['STUSPS', 'Name', 'domain']:
        for z in hwm[group].unique():
            subset = hwm[hwm[group] == z]
            subset = subset[~subset['height_above_gnd_m'].isna()]
            ss = calc_stats(observed=subset['height_above_gnd_m'], modeled=subset['sfincs_hmax_m'])
            ss.append(z)
            ss = pd.DataFrame(ss).T
            ss.columns = ['mae', 'rmse', 'bias', 'group']
            stats_by_group = pd.concat([stats_by_group, ss], axis=0, ignore_index=True)
    stats_by_group.dropna(axis=0, inplace=True)
    stats_by_group.set_index('group', inplace=True, drop=True)
    stats_by_group.to_csv(os.path.join(out_dir, 'hwm_stats_huc6_depth.csv'), index=True)
