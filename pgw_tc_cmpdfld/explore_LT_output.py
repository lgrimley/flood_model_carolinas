import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs')

mod = SfincsModel('LT_2018_ERA5', mode='r')
mod.read_results()