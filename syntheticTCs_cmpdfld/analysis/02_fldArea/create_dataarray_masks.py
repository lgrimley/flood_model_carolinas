import sys
import os
import hydromt
import xarray as xr
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')



wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS'
os.chdir(wdir)

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml, yml_base])

# Read in the data catalog to get the model and basin geom
basins = cat.get_geodataframe(data_like=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\downscale_test\masks\huc6_basins.shp')
basins = basins.to_crs(epsg=32617)

res = 5

'''' Create basin and water body masks  '''

elevation_file = rf'..\03_MODEL_RUNS\subgrid\dep_subgrid_{res}m.tif'
water_mask = rf'.\masks\water_mask_sbgRes{res}m.tif'
basin_mask = rf'.\masks\basin_mask_sbgRes{res}m.tif'

# Load in the elevation file (grid or subgrid)
dep = cat.get_rasterdataset(elevation_file)

# Create a basin mask raster at the grid resolution
mask = dep.raster.rasterize(basins, 'index', nodata=-128, all_touched=False)
mask.rio.write_crs('EPSG:32617', inplace=True)
mask.raster.to_raster(basin_mask, nodata=-128, dtype='int8')

# Now create a mask for the water bodies - coastal and large rivers
# Mask out the cells that are considered water bodies
coastal_wb = cat.get_geodataframe('carolinas_coastal_wb', geom=basins)
coastal_wb = coastal_wb.to_crs(32617)
coastal_wb_clip = coastal_wb.clip(basins)
coastal_wb_clip['mask'] = 1
mask1 = dep.raster.rasterize(coastal_wb_clip, "mask", nodata=0, all_touched=True)

carolinas_nhd_area_rivers = cat.get_geodataframe('carolinas_nhd_area_rivers', geom=basins)
carolinas_nhd_area_rivers = carolinas_nhd_area_rivers.to_crs(32617)
carolinas_nhd_area_rivers['mask'] = 1
mask2 = dep.raster.rasterize(carolinas_nhd_area_rivers, "mask", nodata=0, all_touched=True)

mask = (mask1 + mask2).compute()
mask = xr.where(cond=mask > 0, x=1, y=0)
mask = mask.astype('int8')
mask.rio.write_crs('EPSG:32617', inplace=True)
mask.raster.to_raster(water_mask, nodata=0)


