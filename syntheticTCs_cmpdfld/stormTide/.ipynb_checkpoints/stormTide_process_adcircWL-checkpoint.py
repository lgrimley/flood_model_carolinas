import os
import h5py
import dask.array as da
import matplotlib.pyplot as plt
import scipy.io as sio
import xarray as xr

from syntheticTC_utils import *

# Connect to working directory and load data
# os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\stormTide')
# track_file = r'..\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\CMIP6_585')
stormTide_dir = os.path.join(os.getcwd(), 'stormTide')
track_dir = os.path.join(os.getcwd(), 'tracks')
wl_gage_ids = pd.read_csv(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
scenario = 'ssp585'

stormTide_files = []
for f in os.listdir(stormTide_dir):
    if f.endswith('.mat'):
        stormTide_files.append(f)
gcm_runs = [x.split('_')[0] for x in stormTide_files]

for gcm in gcm_runs:
    # Load the ADCIRC storm tide mat file
    stormTide_file = os.path.join(stormTide_dir, f'{gcm}_{scenario}_WLseries.mat')
    wl_data_mat = h5py.File(stormTide_file)

    # Load the storm tracks
    track_file = os.path.join(track_dir, f'UScoast6_AL_{gcm}_{scenario}cal_roEst1rmEst1_trk100')
    tc_tracks = sio.loadmat(f'{track_file}.mat')

    ''' 

    WLseries.mat description 
    
    WL_inds
    - each row corresponds to an ADCIRC gage output (e.g., 355 stations)
    - the columns list the Storm IDs that are relevant for that gage location
    
    WL_mat
    - shape(355, 2000, 447) = (station, storms, ADCIRC water level output timesteps)
    - the rows are timeseries for a single TC storm at that gage
    - each row corresponds to the Storm ID in the WL_inds data (e.g., row 1 is Storm ID in column 1)
    
    '''

    # Lazy loading of the HDF5 datasets using dask
    print(wl_data_mat.keys())
    wl_inds = da.from_array(wl_data_mat['WL_inds'])
    wl_mat = da.from_array(wl_data_mat['WL_mat'])

    # Subset the data to the gages we are interested in
    sta_ids = wl_gage_ids['gage_id'].values  # gage index
    sta_storms = wl_inds[:, sta_ids-1].compute()  # pull the storms for the gages only (-1 because of python indexing)
    sta_storms_df = pd.DataFrame(sta_storms)  # convert to a dataframe
    sta_storms_df.columns = sta_ids
    sta_storms_df.replace(to_replace=0, value=np.nan, inplace=True)
    sta_storms_df.to_csv(os.path.join(stormTide_dir,f'WL_inds_subset_{gcm}_{scenario}.csv'), index=0)

    # get the unique storm IDs across all the gages
    storms, counts = np.unique(sta_storms_df, return_counts=True)
    storms = pd.DataFrame(storms.transpose(), columns=['tc_id'])
    storms['gage_counts'] = counts.transpose()
    storms.dropna(axis=0, inplace=True)
    storms.to_csv(os.path.join(stormTide_dir, f'stormTide_TCIDs_and_gageCounts_{gcm}_{scenario}.csv'),index=0)
    print(len(storms[storms['gage_counts'] >= 11]))

    # Plot a histogram of the number of storms with various number of WL gages
    fig, ax = plt.subplots(figsize=(4.5, 3), tight_layout=True)
    storms['gage_counts'].hist(ax=ax, bins=22, alpha=0.6, grid=False)
    ax.set_xlim(1, 22)
    ax.set_xticks(ticks=np.arange(1, 22, 2))
    ax.set_xlabel('No. of ADCIRC WL gages')
    ax.set_ylabel('No. of Storms')
    plt.savefig(os.path.join(stormTide_dir,f'Storms_per_AdcircWLgages_{gcm}_{scenario}.png'))
    plt.close()

    # Create a directory to write output
    outdir = os.path.join(stormTide_dir, f'adcirc_waterlevel_netcdf_{gcm}_{scenario}')
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)

    # Set up a dataframe to save the peak water levels at each gage for each storm
    gage_peaks_df = pd.DataFrame(data=sta_ids, columns=['gage_ids'])
    gage_peaks_df.set_index(keys='gage_ids', drop=False, inplace=True)

    # Loop through the storms and pull the gage water levels
    counter = 1
    for tc_id in storms['tc_id']:
        tc_id = int(tc_id)

        fileout = os.path.join(outdir, f'{str(tc_id).zfill(4)}.nc')
        if os.path.exists(fileout) is True:
            print(f'Exists already: {fileout}')
            continue

        # Get the gage IDs relevant for a given storm
        storm_gages = sta_storms_df.columns[sta_storms_df.isin([tc_id]).any()]

        # The index location in the wl_mat file isn't uniform for a given storm
        # we loop through to get that positioning from the wl_ind and then pull the WL
        storm_wl = pd.DataFrame()
        for sg in storm_gages:
            sg_df = sta_storms_df[sg]
            mat_index = sg_df.index.get_loc(sg_df[sg_df == tc_id].index[0])

            # Get the water level timeseries at the selected gages
            sta_wl = wl_mat[:, mat_index, sg].compute()
            sta_wl = pd.DataFrame(sta_wl, columns=[sg])
            storm_wl = pd.concat(objs=[storm_wl, sta_wl], axis=1, ignore_index=True)
        storm_wl.columns = storm_gages

        # Get the storm track date time information
        track_df = get_track_info_in_df(tc_id=tc_id, tc_tracks=tc_tracks)
        track_df = track_df[track_df['datetime'] != 0]
        ref_time = track_df['datetime'][0]

        # use the reference time to calculate the model time at each output timestep
        # Add the datetime information to the dataframe
        storm_wl['datetime'] = [ref_time + datetime.timedelta(hours=t) for t in range(len(storm_wl))]
        storm_wl.set_index(keys='datetime', drop=True, inplace=True)

        # If all the water levels are zero, assume simulation ended and chop'em
        storm_wl = storm_wl.loc[(storm_wl != 0).any(axis=1)]

        # Set bad water levels to np.nan and then linearly interpolate
        storm_wl[(storm_wl < -100.0)] = np.nan
        storm_wl_interp = storm_wl.interpolate(method='linear', axis=1, limit_direction='both')

        # Get the gage peaks for the storm and save to master dataframe
        gage_peaks = pd.DataFrame(storm_wl_interp.max())
        gage_peaks.columns = [tc_id]
        gage_peaks_df = pd.concat(objs=[gage_peaks_df, gage_peaks], axis=1, ignore_index=False)

        # Pull the storm gage x,y info
        coords = wl_gage_ids[wl_gage_ids['gage_id'].isin(storm_gages)]
        if (storm_wl.columns == coords['gage_id']) is False:
            print(f'Mismatch for {tc_id}')
        else:
            # Metadata for writing out the water levels to netcdf
            group = f'{gcm}_{scenario}'
            meta = {'storm_id': tc_id,
                    'group': group,
                    'description': 'ADCIRC water levels for synthetic TC storms',
                    'author': 'Lauren Grimley',
                    'contact': 'lauren.grimley@unc.edu',
                    'track_file': os.path.basename(track_file),
                    'model_data_doi': 'https://www.nature.com/articles/s41558-021-01272-7',
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
            ds_out = xr.Dataset(data_vars=dict(waterlevel=(["time", "index"], storm_wl), ),
                                coords=dict(x=('index', coords['x'].values),
                                            y=('index', coords['y'].values),
                                            index=storm_wl.columns,
                                            time=storm_wl.index.values
                                            ),
                                attrs=meta
                                )
            ds_out.to_netcdf(fileout)
            print(f'{counter} out of {len(storms)} storms processed.')
            counter += 1

        if len(storm_gages) > 20:
            # Plot water level
            fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
            storm_wl.plot(ax=ax, legend=False)
            ax.set_xlabel('')
            ax.set_ylabel('Water Level (m+MSL)')
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f'TC: {tc_id}')
            plt.savefig(os.path.join(outdir, f'{str(tc_id).zfill(4)}_waterlevel.png'))
            plt.close()

    gage_peaks_df.drop(columns='gage_ids', inplace=True, axis=1)
    gage_peaks_df = gage_peaks_df.transpose()
    gage_peaks_df.to_csv(os.path.join(stormTide_dir,f'gage_peaks_{gcm}_{scenario}.csv'))
