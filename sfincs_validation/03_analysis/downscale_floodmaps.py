import os
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils, plots
import xarray as xr
import rioxarray as rio


"""
SCRIPT:
DESCRIPTION:
This script processes SFINCS peak water level and downscales the depth using an input elevation file.

AUTHOR: Lauren Grimley
CONTACT: lauren.grimley@unc.edu

"""


# Filepath to data catalog yml
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml_sfincs_Carolinas)

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model')
model_roots = [
    'ENC_200m_sbg5m_avgN_adv1_eff75',
    #'ENC_200m_sbg5m_avgN_adv1_eff75_coastal',
    'ENC_200m_sbg5m_avgN_adv1_eff75_runoff',
    # 'ENC_200m_sbg5m_avgN_adv1_eff75_discharge',
    # 'ENC_200m_sbg5m_avgN_adv1_eff75_rainfall',
    # 'ENC_200m_sbg5m_avgN_adv1_eff75_stormTide',
    # 'ENC_200m_sbg5m_avgN_adv1_eff75_wind'
]

dep_file = os.path.join(os.getcwd(), 'ENC_200m_sbg5m_avgN_adv1_eff50_compound', 'subgrid', 'dep_subgrid.tif')
dep_da = cat.get_rasterdataset(dep_file)
# dep_da = mod.grid['dep']

# Create directories
out_dir = os.path.join(os.getcwd(), 'floodmaps', '5m')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

da_list = []
for model_root in model_roots:
    mod = SfincsModel(model_root, mode='r',
                      data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
    zsmax = mod.results["zsmax"].max(dim='timemax')
    output_file = os.path.join(out_dir, f"{model_root}_linear.tif")

    # Downscale to subgrid
    hmax = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=dep_da,
        hmin=0.05,
        gdf_mask=mod.region,
        reproj_method='bilinear',
        floodmap_fn=output_file
    )
    da_list.append(hmax)
    print('Finished with downscaling scenario: ', model_root)

# da = xr.concat(da_list, dim='run')
# da['run'] = xr.IndexVariable('run', model_roots)
# da.to_netcdf(os.path.join(out_dir, f"floodmaps.nc"))
