#!/usr/bin/env python3
# coding: utf-8

########################################################################################################
#    Author: Lauren E. Grimley
#    Contact: lgrimley@unc.edu
########################################################################################################

from osgeo import gdal
import os


def get_raster_ext(rasterfile):
    file = os.path.basename(rasterfile)
    ext = file.split('.')
    if ext[-1] != 'tif':
        output_file = os.path.join(os.path.dirname(rasterfile), ext[0] + '.tif')
        raster = gdal.Open(rasterfile, gdal.GA_ReadOnly)
        gdal.Translate(destName=output_file, srcDS=raster)
        ras = output_file
    else:
        ras = rasterfile
    print("Using raster file:", os.path.basename(ras))
    return ras


def open_raster(rasterfile):
    raster = gdal.Open(rasterfile)
    return raster


def compute_statistics(raster):
    band = raster.GetRasterBand(1)
    print('Raster values of type:', gdal.GetDataTypeName(band.DataType))
    if band.GetMinimum() is None or band.GetMaximum() is None:
        band.ComputeStatistics(0)
    print("[ NO DATA VALUE ] = ", band.GetNoDataValue())
    print("[ MIN ] = ", band.GetMinimum())
    print("[ MAX ] = ", band.GetMaximum())


def check_projections(f1, f2):
    raster1 = open_raster(f1)
    raster2 = open_raster(f2)
    prj1 = raster1.GetProjection()
    prj2 = raster2.GetProjection()
    if prj1 != prj2:
        print("Projecting Raster:", os.path.basename(f2))
        gdal.Warp(f2, raster2, dstSRS=prj1)
        raster2 = open_raster(f2)
    else:
        print('Raster Projections Match')
    return raster1, raster2


def get_pixel_size(raster):
    gt = raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY = gt[5]
    if abs(pixelSizeX) == abs(pixelSizeY):
        resolution = abs(pixelSizeX)
        print('Target resolution:', resolution)
    else:
        print('Cell X and Y do not match. Check target raster.')
        print('X=', pixelSizeX, 'Y=', pixelSizeY)
    return resolution


def align_ras2_to_ras1(ras1, ras2):
    target_res = get_pixel_size(ras1)
    gdal.Warp(destNameOrDestDS=f2_out, srcDSOrSrcDSTab=ras2,
              targetAlignedPixels=ras1, xRes=target_res, yRes=target_res)
    print('Raster 2 pixels are aligned to Raster 1:', os.path.basename(f2_out))


def geotiff_to_ascii(rasterfile):
    out_file = os.path.basename(rasterfile).split('.')[0]
    output_file = os.path.join(os.path.dirname(rasterfile), out_file + '.asc')
    raster = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    gdal.Translate(destName=output_file, srcDS=raster)
    print('Convert from GeoTiff to ASCII:', os.path.basename(rasterfile))
    return output_file


def build_pyramids(rasterfile):
    raster = gdal.Open(rasterfile, 0)  # 0 = read-only, 1 = read-write
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    raster.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
    print('Building pyramids:', os.path.basename(rasterfile))



# ------------- User Input ------------------
filepath1 = "/Users/laurengrimley/Desktop/model_input_10m/dem_10m_msk"
filepath2 = "/Users/laurengrimley/Desktop/model_input_10m/rough_30m.asc"


# ------------- Execute Functions ------------------
# Get rasters and convert to GeoTiff
f1 = get_raster_ext(filepath1)
f2 = get_raster_ext(filepath2)

# Open rasters and make sure the projections are the same
ras1, ras2 = check_projections(f1, f2)

# Align raster 2 to raster 1
f2_out = os.path.basename(f2).split(".")
f2_out = os.path.join(os.path.dirname(f2), f2_out[0] + "_aligned.tif")
align_ras2_to_ras1(ras1, ras2)

# Convert from GeoTiff to ASCII Grids
grid1 = geotiff_to_ascii(filepath1)
grid2 = geotiff_to_ascii(f2_out)

# Build Pyramids
build_pyramids(grid1)
build_pyramids(grid2)


