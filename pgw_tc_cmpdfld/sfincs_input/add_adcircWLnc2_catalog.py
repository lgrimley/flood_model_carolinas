import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils
import shutil

# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley
# Last updated: 08/21/2023

root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input'
data_lib = os.path.join(root, 'data_catalog_pgw.yml')
cat = hydromt.DataCatalog(data_lib)


storms = ['floyd', 'florence', 'matthew']
climates = ['future']

for storm in storms:
    if storm == 'floyd':
        ss = 'floy'
    elif storm == 'florence':
        ss = 'flor'
    elif storm == 'matthew':
        ss = 'matt'

    for climate in climates:
        if climate == 'present':
            cc = 'pres'
        elif climate == 'future':
            cc = 'fut'

        event = f'{climate}_{storm}'
        rename = f'{ss}_{cc}'
        dir = os.path.join(root, 'waterlevel', event)
        for filename in os.scandir(dir):
            run = filename.name.split('_')[-1].split('.')[0]
            shutil.move(src=filename.path, dst=os.path.join(dir, f'{rename}_{run}.nc'))
            name = f'{rename}_{run}_waterlevel'
            # path_dc = f'waterlevel/{event}/{rename}_{run}.nc'
            yml_str = f"""

# {name}:
#     path: {path_dc}
#     data_type: GeoDataset
#     driver: netcdf
#     crs: 4326
#     meta:
#         category: waterlevel"""

            with open(data_lib, mode="a") as f:
                f.write(yml_str)
            print('Done writing dataset to data catalog')

