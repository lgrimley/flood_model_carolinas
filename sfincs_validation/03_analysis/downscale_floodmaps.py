import os
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils, plots
import xarray as xr
import rioxarray as rio


os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model')

scenarios = [
    # 'ENC_200m_sbg5m_avgN_adv1_eff50_coastal',
    # 'ENC_200m_sbg5m_avgN_adv1_eff50_runoff',
    'ENC_200m_sbg5m_avgN_adv1_eff50_compound',
]
hmin = 0.05

for scen in scenarios:
    mod = SfincsModel(scen, mode='r')
    sbgdep = mod.data_catalog.get_rasterdataset(
        os.path.join(os.getcwd(), 'ENC_200m_sbg5m_avgN_adv1_eff50_compound', 'subgrid', 'dep_subgrid.tif'))
    zsmax = mod.results["zsmax"].max(dim='timemax')

    # Downscale to subgrid
    hmax = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=sbgdep,
        hmin=hmin,
        gdf_mask=None,
        reproj_method='bilinear',
        floodmap_fn=os.path.join(mod.root, f"compound_hmax_sbg5m_hmin_5cm_v2.tif")
    )
    # fldmaps_ls.append(hmax)
    print('Finished with downscaling scenario: ', scen)

# da = xr.concat(fldmaps_ls, dim='run')
# da['run'] = xr.IndexVariable('run', scenarios)
# da.to_netcdf("floodmap_" + str(hmin) + "_hmin.nc")
