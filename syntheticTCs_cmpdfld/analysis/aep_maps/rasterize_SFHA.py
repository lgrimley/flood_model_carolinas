import os
import numpy as np
from hydromt_sfincs import SfincsModel
import geopandas as gpd
import xarray as xr

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS')

# Load SFINCS model
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
mod.read_grid()

# Load model DEM
dem_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\subgrid\dep_subgrid_20m.tif'
dem_20m = mod.data_catalog.get_rasterdataset(dem_filepath)
dem = mod.grid['dep']

# Load NC SFHA shapefile and rasterize
nc_sfha = gpd.read_file(filename=r'Z:\Data-Expansion\users\lelise\data\geospatial\FEMA\NFHL_37_20250205\NFHL_37_20250205.gdb',
                        layer='S_FLD_HAZ_AR',
                        mask=mod.region).to_crs(mod.crs)
nc_sfha['code_mask'] = -9999
zones = nc_sfha['FLD_ZONE'].unique()
print(zones)
codes = np.arange(0, len(zones))
zone_codes = [f'{zones[i]}={codes[i]}' for i in range(len(zones))]
condensed_string = ", ".join(zone_codes)
for i in range(len(zones)):
    nc_sfha.loc[nc_sfha['FLD_ZONE'] == zones[i], 'code_mask'] = i
nc_sfha_r_200m = dem.raster.rasterize(nc_sfha, "code_mask", nodata=-9999, all_touched=False)
nc_sfha_r_200m.attrs = {'Codes' : condensed_string}
#nc_sfha_r_20m = dem_20m.raster.rasterize(nc_sfha, "code_mask", nodata=-9999, all_touched=False)
#nc_sfha_r_20m.attrs = {'Codes' : condensed_string}


# Repeat for south carolina
sc_sfha = gpd.read_file(filename=r'Z:\Data-Expansion\users\lelise\data\geospatial\FEMA\NFHL_45_20241103\NFHL_45_20241103.gdb',
                        layer='S_FLD_HAZ_AR',
                        mask=mod.region).to_crs(mod.crs)
sc_sfha['code_mask'] = -9999
zones = sc_sfha['FLD_ZONE'].unique()
codes = np.arange(0, len(zones))
zone_codes = [f'{zones[i]}={codes[i]}' for i in range(len(zones))]
condensed_string = ", ".join(zone_codes)
for i in range(len(zones)):
    sc_sfha.loc[sc_sfha['FLD_ZONE'] == zones[i], 'code_mask'] = i
sc_sfha_r_200m = dem.raster.rasterize(sc_sfha, "code_mask", nodata=-9999, all_touched=False)
sc_sfha_r_200m.attrs = {'Codes' : condensed_string}
#sc_sfha_r_20m = dem_20m.raster.rasterize(sc_sfha, "code_mask", nodata=-9999, all_touched=False)
#sc_sfha_r_20m.attrs = {'Codes' : condensed_string}

# Write to netcdf
m1 = nc_sfha_r_200m > 0
m2 = sc_sfha_r_200m > 0
sfha_200m2 = xr.where(cond=nc_sfha_r_200m < 0, x=sc_sfha_r_200m, y=nc_sfha_r_200m)
print(np.unique(sfha_200m2))
sfha_200m2 = xr.where(cond=sfha_200m2 ==9999, x=sfha_200m2, y=np.nan)
sfha_200m2.attrs = sc_sfha_r_20m.attrs
sfha_200m2['spatial_ref'] = dem['spatial_ref']
sfha_200m2.plot()
sfha_200m2.to_netcdf('Carolinas_SFHA_200m.nc')

m1 = nc_sfha_r_20m > 0
m2 = sc_sfha_r_20m > 0
sfha_20m2 = (nc_sfha_r_20m.where(m1) + sc_sfha_r_20m.where(m2)).compute()
sfha_20m2.to_netcdf('Carolinas_SFHA_20m.nc')
