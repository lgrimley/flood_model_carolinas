#!/usr/bin/env python3
# coding: utf-8
########################################################################################################
#    Author: Lauren E. Grimley
#    Contact: lgrimley@unc.edu
########################################################################################################
#import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import numpy as np
import os


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


hwm_f = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/scripts/FilteredHWMs.csv'
mxe_f = 'C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output/florence-m5.mxe'
#max_f = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/output/florence-m1.max'

# HWM
df = pd.read_csv(hwm_f, delimiter=',', usecols=['longitude', 'latitude', 'elev_ft'])
df['elev_ft'] = df['elev_ft'] * 0.3048
hwm = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude, z=df.elev_ft, crs=4326))
hwm = hwm.to_crs('EPSG:32119')
coords = [(x, y) for x, y in zip(hwm.geometry.x, hwm.geometry.y)]

raster_in = gdal.Open(mxe_f)
raster_out = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/scripts/florence_m5_mxe.tif'
gdal.Warp(raster_out, raster_in, dstSRS='EPSG:32119', srcSRS='EPSG:32119')
# os.system(r'gdalwarp -t_srs "EPSG:4269" -r bilinear "%s" "%s"' % (raster_in, raster_out))

# Load and Plot WSE raster
src = rasterio.open(raster_out, 'r')
plt.imshow(raster_in)

# Extract WSE at HWM Locations

test = [x[0] for x in src.sample(xy=coords, masked=True)]
mask = hwm.mask(cond=np.array(test), inplace=True)


test = hwm[hwm.where(hwm['modeled'] != -9999.0)]

hwm['diff'] = hwm['elev_ft'] - hwm['modeled']

test = gpd.clip(gdf=hwm, mask=hwm.mask, keep_geom_type=True)

hwm.plot(column='diff')
