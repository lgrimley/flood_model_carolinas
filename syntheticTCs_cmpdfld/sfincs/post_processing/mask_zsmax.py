import os
from hydromt_sfincs import SfincsModel
from scipy import ndimage
import rasterio

def resample_raster(src_raster, target_resolution, method='nearest'):
    """
    Resample a raster to a new resolution.

    :param src_raster: Path to the source raster file (lower resolution).
    :param target_resolution: The target resolution (higher resolution).
    :param method: The interpolation method (options: 'nearest', 'bilinear', 'cubic').
    :return: Resampled raster data.
    """
    with rasterio.open(src_raster) as src:
        # Read the raster data
        src_data = src.read(1)  # assuming single band

        # Calculate the scaling factor
        scale_factor = src.res[0] / target_resolution

        # Create an output shape based on the scaling factor
        new_shape = (int(src_data.shape[0] * scale_factor), int(src_data.shape[1] * scale_factor))

        # Use scipy's zoom function to resample the raster
        resampled_data = ndimage.zoom(src_data, zoom=(scale_factor, scale_factor), order=3 if method == 'cubic' else 1)

        return resampled_data

# Load SFINCS model for importing results
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
cat = sfincs_mod.data_catalog

os.chdir(r'Z:\Data-Expansion\users\lelise\data_share\SFINCS_OUTPUT\published_on_NHERI_120324')

raster1_path = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif'  # Higher resolution raster
raster2_path = r'.\sfincs wse\SFINCS_FloydHindcast_MaxWSE.tif'  # Lower resolution raster
# Read the first raster (higher resolution)
with rasterio.open(raster1_path) as src1:
    raster1_data = src1.read(1)  # Read the first band
    raster1_transform = src1.transform  # Affine transform for georeferencing
    raster1_resolution = src1.res[0]
raster2_resampled = resample_raster(raster2_path, raster1_resolution, method='nearest')

if raster1_data.shape != raster2_resampled.shape:
    raise ValueError("The rasters do not have the same shape. Make sure they have the same resolution and extent.")

output_raster_path = '.\sfincs wse\SFINCS_FloydHindcast_MaxWSE_20m.tif'
# Create the output raster with the same georeferencing as the first raster
with rasterio.open(output_raster_path, 'w', driver='GTiff', count=1, dtype=raster2_resampled.dtype,
                   width=raster1_data.shape[1], height=raster1_data.shape[0], crs=src1.crs,
                   transform=raster1_transform) as dst:
    dst.write(raster2_resampled, 1)

# Subgrid elevation
dep_sbg = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')
zsmax_sbg = cat.get_rasterdataset(r'.\sfincs wse\SFINCS_FloydHindcast_MaxWSE_20m.tif')
mask = zsmax_sbg > dep_sbg
zsmax_out = zsmax_sbg.where(mask)
zsmax_out.raster.to_raster(r'.\sfincs wse 20m\floyd_wse_mask_20m.tif', nodata=-9999.0)

# Water level greater than elevation at the grid resolution
# zsmax = cat.get_rasterdataset(r'.\sfincs wse\SFINCS_FlorenceHindcast_MaxWSE.tif')
# dep = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep.tif')
# mask = zsmax > dep
# zsmax_out = zsmax.where(mask)
#zsmax_out.raster.to_raster('florence_wse_mask.tif', nodata=-9999.0)