import pandas as pd
import os
import xarray as xr

rain_dir = r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\rain'
tracks_dir = r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\tracks'

track_files = ['UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100_200km',
                 'adcirc_modeled_TCs_all',
                 'adcirc_modeled_TCs_outside200kmBuffer',
                 'noSurge_TCs_within200kmBuffer']

for f in track_files:

    tc_d = pd.read_csv(os.path.join(tracks_dir, f'{f}.csv'))
    total_precip_grids = []
    tc_rain_info = pd.DataFrame()

    n_tc = len(tc_d['tc_id'])
    for i in range(n_tc):
        tc_id = tc_d['tc_id'][i]
        grid = f'{str(tc_id).zfill(4)}.nc'
        d = xr.open_dataset(os.path.join(rain_dir, '03_TCR_RainOutput_Gridded',grid))
        total_precip = d.sum(dim='time')
        total_precip_grids.append(total_precip)

        # Calculate storm stats for precip and output
        max_total_precip_cell = round(d.sum(dim='time').max()['precip'].item(), 3)
        cumm_precip_domain = round(d.sum()['precip'].item(), 3)
        max_rain_rate = round(d.max(dim='time').max()['precip'].item(), 3)
        x = pd.DataFrame([tc_id, max_total_precip_cell, cumm_precip_domain, max_rain_rate]).T
        tc_rain_info = pd.concat(objs=[tc_rain_info, x], axis=0)
        print(f'{i} out of {n_tc}')

    outdir = os.path.join(rain_dir, f)
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)

    tc_rain_info.columns = ['tc_id', 'max_total_precip_cell', 'cumm_precip_domain', 'max_rain_rate']
    tc_rain_info.to_csv(os.path.join(outdir, 'tc_rain_info.csv'), index=False)

    da = xr.concat(total_precip_grids, dim='run')
    da['run'] = xr.IndexVariable('run', tc_rain_info['tc_id'].values)
    da.to_netcdf(os.path.join(outdir,'TC_TotPrecip_grids.nc'))
