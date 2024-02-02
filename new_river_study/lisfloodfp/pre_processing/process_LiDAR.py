#!/usr/bin/env python3
# coding: utf-8

import os
import geopandas as gpd
from descartes.patch import PolygonPatch
import rasterio
from rasterio.merge import merge
from rasterio.plot import show


def project_shapefile(filepath):
    shapefile = gpd.read_file(filepath)
    shapefile.to_crs(epsg=4326)
    return shapefile


def which_lidar_tiles(basin_filepath, lidar_tiles_filepath):
    lidar_tiles = project_shapefile(lidar_tiles_filepath)
    basin = project_shapefile(basin_filepath)
    tiles_subset = gpd.overlay(df1=lidar_tiles, df2=basin, how='intersection')
    tiles_subset.plot(alpha=0.5, edgecolor='k')
    return tiles_subset


def get_lidar_filepaths(dem_resolution, tiles):
    dem_col = 'DEM' + str(str(dem_resolution).zfill(2)) + '_Path'
    filepaths = [os.path.normpath(os.path.join(*os.path.split(x)[0].split('\\')[-4:])) for x in tiles[dem_col]]
    filenames = [os.path.basename(x) for x in tiles[dem_col]]
    f = list()
    for x in range(len(filenames)):
        f.append(os.path.join(filepaths[x], filenames[x]))
    return f


tiles_f = 'C:/Users/lelise/UNC_Research/NewRiver/gis/topobathy/QL1_QL2_Tile_Layout_w_size/QL2_QL1_Tiling_Scheme.shp'
basin_f = 'C:/Users/lelise/UNC_Research/NewRiver/gis/commondata/nr/watershed/NR_watershed_prj.shp'
dir = 'C:/Users/lelise/UNC_Research/Data/lidar/NC_LIDAR_5m/'

lidar_files = get_lidar_filepaths(dem_resolution=5,
                                  tiles=which_lidar_tiles(basin_f, tiles_f))

f = list()
for x in range(len(lidar_files)):
    f.append(os.path.join(dir, lidar_files[x]))

src_files_to_mosaic = []
for file in f:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)

mosaic, out_transform = merge(datasets=src_files_to_mosaic)
show(mosaic)

out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "})