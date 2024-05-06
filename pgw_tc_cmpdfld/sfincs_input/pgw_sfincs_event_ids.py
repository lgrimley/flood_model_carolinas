import os
import datetime
import hydromt
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib as mpl
import cartopy.crs as ccrs
import scipy.stats as ss

storms = ['floy', 'matt', 'flor']
climates = ['pres', 'fut']
scenarios = ['compound', 'runoff', 'coastal']
slr_event_db = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\slr_event_ids.csv',
                           header=None)
slr_event_db.columns = ['event_id', 'slr']


def get_event_ids(storms, climates, slr_events, scenarios):
    event_id_list = []
    # for storm in storms:
    #     runs = ['ensmean', 'ens1', 'ens2', 'ens3', 'ens4', 'ens5', 'ens6', 'ens7']
    #     if storm == 'flor':
    #         # Florence
    #         tref = '20180913 000000'
    #         tstart = tref
    #         tstop = '20180930 000000'
    #     elif storm == 'matt':
    #         # Matthew
    #         tref = '20161007 000000'
    #         tstart = tref
    #         tstop = '20161015 000000'
    #         runs = ['ensmean', 'ens1', 'ens2', 'ens3', 'ens4', 'ens5', 'ens6']
    #     elif storm == 'floy':
    #         # Floyd
    #         tref = '19990913 000000'
    #         tstart = '19990914 000000'
    #         tstop = '19990922 000000'
    #     for climate in climates:
    #         for run in runs:
    #             for scen in scenarios:
    #                 event_id = f'{storm}_{climate}_{run}_{scen}'
    #                 event_id_list.append(event_id)
    for id in slr_events:
        for scen in scenarios:
            event_id = f'{id}_{scen}'
            event_id_list.append(event_id)
    return event_id_list


event_ids = get_event_ids(storms=storms, climates=climates,
                          slr_events=slr_event_db['event_id'], scenarios=scenarios)

event_ids = pd.DataFrame(event_ids)
event_ids.columns = ['event_id']

event_ids.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\sl_event_ids_all.csv',
                 header=False, index=False)
