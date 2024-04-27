import os
import datetime
import hydromt
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib as mpl
import cartopy.crs as ccrs
import scipy.stats as ss

# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'

# Working directory and model root
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\present_matthew\ensmean\matt_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])
mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\tmp')
mod.write()

# Creating meteo inputs from present scaled scenario
sf_df = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors'
                    r'\presScaled_precip_multiplier.csv',
                    header=None)
sf_df.columns = ['event_id', 'multiplier']
sf_df.set_index('event_id', drop=True, inplace=True)

wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]
nrow = 1
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol - 1)
last_row = np.arange(ncol, n_subplots, 1)

dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models', 'precip2d')
for filename in os.listdir(dir_out):
    storm, climate, run, variable = filename.split('_')
    if climate == 'pres':
        multiplier = sf_df.loc[f'{storm}_{run}'][0]
        da = mod.data_catalog.get_rasterdataset(os.path.join(dir_out, filename))
        da_scaled = da * multiplier
        fileout = filename.replace('pres', 'presScaled')
        # da_scaled.to_netcdf(os.path.join(dir_out, fileout))

        ''' Meteo Plots '''
        precip_bounds = np.arange(100, 1000, 50)
        plot_total_meteo = True
        if plot_total_meteo is True:
            fig, axs = plt.subplots(
                nrows=nrow, ncols=ncol,
                figsize=(8, 5),
                tight_layout=True,
                subplot_kw={'projection': utm},
                layout='constrained',
                sharex=True, sharey=True)

            cmap = mpl.cm.jet
            norm = mpl.colors.BoundaryNorm(precip_bounds, cmap.N, extend='both')
            da.sum(dim='time').plot(ax=axs[0],
                                    cmap=cmap,
                                    norm=norm,
                                    add_colorbar=False,
                                    zorder=2,
                                    alpha=0.8)
            axs[0].set_title('')
            axs[0].set_title(f'present {storm} {run}', loc='center')

            da_scaled.sum(dim='time').plot(ax=axs[1],
                                           cmap=cmap,
                                           norm=norm,
                                           add_colorbar=False,
                                           zorder=2,
                                           alpha=0.8)
            axs[1].set_title('')
            axs[1].set_title(f'rain rates multiplier: {round(multiplier, 2)}', loc='center')

            pos1 = axs[1].get_position()  # get the original position
            cbar_ax = fig.add_axes([pos1.x1 + 0.02, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cb = fig.colorbar(sm,
                              cax=cbar_ax,
                              shrink=0.7,
                              extend='both',
                              spacing='uniform',
                              label='Total Precipitation\n(mm)',
                              pad=0,
                              aspect=10)
            axs = axs.flatten()

            for ii in range(len(axs)):
                # minx, miny, maxx, maxy = extent.total_bounds
                # axs[ii].set_xlim(minx, maxx)
                # axs[ii].set_ylim(miny, maxy)
                axs[ii].set_extent(extent, crs=utm)
                mod.region.plot(ax=axs[ii], color='none', edgecolor='black', linewidth=1,
                                linestyle='-', zorder=2, alpha=1, label='HUC6 Basins')

                # Plot background/geography layers
                if ii in first_in_row:
                    axs[ii].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
                    axs[ii].yaxis.set_visible(True)
                    axs[ii].ticklabel_format(style='sci', useOffset=False)
                elif ii in last_row:
                    axs[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
                    axs[ii].xaxis.set_visible(True)
                    axs[ii].ticklabel_format(style='sci', useOffset=False)

            # Save and close plot
            plt.subplots_adjust(wspace=0.05, hspace=0.1)
            plt.margins(x=0, y=0)

            dir_fig = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\forcing_figs')
            if os.path.exists(dir_fig) is False:
                os.makedirs(dir_fig)
            figout = os.path.join(dir_fig, f'{storm}_{run}_scaled.png')
            plt.savefig(figout, bbox_inches='tight', dpi=255)
            plt.close()

# Sea level rise
slr_df = pd.DataFrame()
dir = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\slr_projections\ipcc_ar6')
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(dir, filename))
        slr_df = pd.concat([slr_df, df])
        print(filename)
slr_df.reset_index(inplace=True, drop=True)

slr_sub = slr_df[(slr_df['scenario'] == 'ssp585') & (slr_df['confidence'] == 'medium')]
stations = slr_sub['psmsl_id'].unique()
sta_data = pd.DataFrame()
for sta in stations:
    percentiles = slr_sub[slr_sub['psmsl_id'] == sta]['2100']

    # Determine the parameters (minimum and maximum) of the uniform distribution
    a = min(percentiles)
    b = max(percentiles)
    uniform_sample = ss.uniform.rvs(loc=a, scale=b - a, size=1000)

    # Plot the empirical cumulative distribution function (ECDF)
    sorted_data = np.sort(uniform_sample)
    sta_data[f'{sta}'] = sorted_data
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

sta_data.columns = ['Springmaid Pier SC', 'Cape Hatteras NC', 'Beaufort NC', 'Oregon Inlet NC']
sta_data.index = ecdf

fig, ax = plt.subplots()
plt.plot(sta_data, sta_data.index, lw=2, label=sta_data.columns)
plt.xlabel('Sea Level Rise (m)')
plt.ylabel('Cumulative Probability')
plt.title('IPCC AR6 projections for SSP5-8.5 in 2100 (~4 degC warming)\nrelative to 1995-2014 baseline')
plt.legend()
plt.grid(True)
plt.savefig(fr'Z:\users\lelise\projects\ENC_CompFld\slr_projections\ipcc_ar6\ipcc_ar6_ssp585_medium.png',
            tight_layout=True)
plt.close()

# Sample random and apply to model water level BCs
percentiles = slr_sub[slr_sub['psmsl_id'] == 2295]['2100']
a = min(percentiles)
b = max(percentiles)
randsamp = ss.uniform.rvs(loc=a, scale=(b - a), size=10)

dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\waterlevel')
for filename in os.listdir(dir_out):
    storm, climate, run, variable = filename.split('_')
    if climate == 'pres':
        multiplier = sf_df.loc[f'{storm}_{run}'][0]
        da = mod.data_catalog.get_rasterdataset(os.path.join(dir_out, filename))
        da_scaled = da * multiplier
        fileout = filename.replace('pres', 'presScaled')
        # da_scaled.to_netcdf(os.path.join(dir_out, fileout))
