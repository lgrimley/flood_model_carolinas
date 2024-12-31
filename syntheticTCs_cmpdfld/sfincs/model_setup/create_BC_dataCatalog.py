import os

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585')
data_lib = 'data_catalog_NCEP.yml'

data_dir = r'.\rain\TCR_Gridded_canesm_ssp585cal_hourly'
with open(data_lib, mode="w+") as fcat:
    for file in os.listdir(data_dir):
        if file.endswith('.nc'):
            tc_id = file.split('.')[0]
            yml_str = f"""
precip_{tc_id}:
  path: {os.path.join(data_dir, file)}
  data_type: RasterDataset
  driver: netcdf
  crs: 4326
  meta:
    category: meteo
    units: mm/hr
    description: hourly gridded rain rates for synthetic tropical cyclone tracks
    title: Synthetic TCR rainfall
    model_ref: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013GL058284
    publication_date: 2022
    model: canesm ssp585cal
    data_ref: https://www.nature.com/articles/s41558-021-01272-7"""

            fcat.write(yml_str)


data_dir = r'.\wind\02_CLE15_WindOutput_Gridded'
with open(data_lib, mode="a") as fcat:
    for file in os.listdir(data_dir):
        if file.endswith('.nc'):
            tc_id = file.split('.')[0]
            yml_str = f"""
wind_{tc_id}:
  path: {os.path.join(data_dir, file)}
  data_type: RasterDataset
  driver: netcdf
  crs: 4326
  meta:
    category: meteo
    units: m/s
    description: hourly gridded wind speeds for synthetic tropical cyclone tracks
    title: TC parametric wind fields
    model_ref: https://journals.ametsoc.org/view/journals/mwre/108/8/1520-0493_1980_108_1212_aamotw_2_0_co_2.xml
    publication_date: 2022
    data_ref: https://www.nature.com/articles/s41558-021-01272-7"""

            fcat.write(yml_str)


data_dir = r'.\stormTide\adcirc_waterlevel_netcdf'
with open(data_lib, mode="a") as fcat:
    for file in os.listdir(data_dir):
        if file.endswith('.nc'):
            tc_id = file.split('.')[0]
            yml_str = f"""
stormTide_{tc_id}:
  path: {os.path.join(data_dir, file)}
  data_type: GeoDataset
  driver: netcdf
  crs: 4326
  meta:
    category: waterlevel
    units: m+MSL
    description: hourly modeled water levels for synthetic tropical cyclone tracks
    title: Synthetic TC ADCIRC StormTide
    model_ref: ADCIRC
    publication_date: 2022
    data_ref: https://www.nature.com/articles/s41558-021-01272-7"""

            fcat.write(yml_str)


data_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data'
file = 'EDSReanalysis_V2_1992_2022.nc'
with open(data_lib, mode="a") as fcat:
    yml_str = f"""
tide_reanalysis:
  path: {os.path.join(data_dir, file)}
  data_type: GeoDataset
  driver: netcdf
  crs: 4326
  meta:
    category: waterlevel
    units: m+MSL
    description: https://github.com/RENCI/EDSReanalysis
    title: ADCIRC tide reanalysis V2
    model_ref: ADCIRC
    publication_date: 2024
    data_ref: https://renci.github.io/edsreanalysisdoc/"""
    fcat.write(yml_str)