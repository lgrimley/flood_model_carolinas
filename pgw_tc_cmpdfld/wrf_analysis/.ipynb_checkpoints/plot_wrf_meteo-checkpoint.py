import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

mod_root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence' \
           r'\future_florence_ensmean'
mod = SfincsModel(mod_root, mode='r')
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\precip2d')
out_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\precip'

# UTM / Figure Extents
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]
nrow = 4
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

# Plotting the total precipitation
totprecip_bounds = np.arange(100, 800, 50)
rrprecip_bounds = np.arange(5, 60, 5)
storms = ['floy', 'matt', 'flor']
climates = ['pres', 'fut', 'presScaled']
for var in ['total_precip', 'max_rain_rate']:
    for climate in climates:
        for storm in storms:
            file_list = []
            for (root, dirs, file) in os.walk(os.getcwd()):
                for f in file:
                    if storm in f and climate == f.split('_')[1]:
                        file_list.append(f)

            fig, axs = plt.subplots(
                nrows=nrow, ncols=ncol,
                figsize=(5.5, 8),
                subplot_kw={'projection': utm},
                tight_layout=True,
                layout='constrained',
                sharex=True, sharey=True)
            axs = axs.flatten()
            for i in range(len(file_list)):
                ax = axs[i]
                cmap = mpl.cm.jet
                precip = mod.data_catalog.get_rasterdataset(file_list[i])
                if var == 'total_precip':
                    norm = mpl.colors.BoundaryNorm(totprecip_bounds, cmap.N, extend='both')
                    precip.sum(dim='time').plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=2, alpha=1)
                    label = 'Total Precipitation\n(mm)'
                elif var == 'max_rain_rate':
                    norm = mpl.colors.BoundaryNorm(rrprecip_bounds, cmap.N, extend='both')
                    precip.max(dim='time').plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=2, alpha=1)
                    label = 'Max Rain Rate\n(mm/hr)'
                ax.set_title('')
                ax.set_title(file_list[i].split('_')[2], loc='center')
            for ii in range(len(axs)):
                minx, miny, maxx, maxy = mod.region.total_bounds
                axs[ii].set_xlim(minx, maxx)
                axs[ii].set_ylim(miny, maxy)
                axs[ii].set_extent(extent, crs=utm)
                mod.region.plot(ax=axs[ii], color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=2,
                                alpha=1)
                if ii in first_in_row:
                    axs[ii].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
                    axs[ii].yaxis.set_visible(True)
                    axs[ii].ticklabel_format(style='sci', useOffset=False)
                if ii in last_row:
                    axs[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
                    axs[ii].xaxis.set_visible(True)
                    axs[ii].ticklabel_format(style='sci', useOffset=False)

            if storm == 'matt':
                axs[-1].set_axis_off()
            # Colorbar - Precip
            pos1 = axs[last_in_row[2]].get_position()  # get the original position
            cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.25, 0.05, pos1.height * 1.5])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cb = fig.colorbar(sm,
                              cax=cbar_ax,
                              shrink=0.7,
                              extend='both',
                              spacing='uniform',
                              label=label,
                              pad=0,
                              aspect=10)
            # Save and close plot
            plt.subplots_adjust(wspace=0.0, hspace=0.15, top=0.94)
            plt.margins(x=0, y=0)
            plt.suptitle(f'{storm} {climate}')
            plt.savefig(os.path.join(out_dir, f'{storm}_{climate}_{var}.png'),
                        bbox_inches='tight', dpi=255)
            plt.close()
