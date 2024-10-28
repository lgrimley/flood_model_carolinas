import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils
import cartopy.crs as ccrs
from shapely.geometry import LineString
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\wrf_tracks')
car_bbox = gpd.read_file('.\wrf_domain_shp\carolinas_bbox_pgw.shp')
flor_bbox = gpd.read_file('.\wrf_domain_shp\wrf_flor_bbox_pgw.shp')
matt_bbox = gpd.read_file('.\wrf_domain_shp\wrf_matt_bbox_pgw.shp')
floy_bbox = gpd.read_file('.\wrf_domain_shp\wrf_floy_bbox_pgw.shp')
wrf_domains = [flor_bbox, floy_bbox, matt_bbox]
usa = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k'
                    r'\cb_2018_us_state_500k.shp')

storm_tracks = []
for storm in ['flor','floy','matt']:
    track_gdfs = pd.DataFrame()
    ii=8
    wd = rf'.\{storm}_ensmean_txt_files'
    if storm =='matt':
        ii=7
    for climate in ['present','future']:
        for i in range(1,ii):
            if climate == 'present':
                tc_id = f'ens{i}'
            else:
                tc_id = f'ens{i}_pgw'

            track_lat = pd.read_table(os.path.join(wd, f'minlat_{tc_id}.txt'), header=None)
            track_lon = pd.read_table(os.path.join(wd, f'minlon_{tc_id}.txt'), header=None)
            track_lat = [item for sublist in track_lat.values.tolist() for item in sublist]
            track_lon = [item for sublist in track_lon.values.tolist() for item in sublist]

            if storm == 'flor':
                track_lat = track_lat[:-4]
                track_lon = track_lon[:-4]

            track_coords = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=track_lon, y=track_lat, crs=4326))
            track_coords['tc_id'] = tc_id
            track_line = track_coords.groupby('tc_id')['geometry'].apply(lambda x: LineString(x.tolist()))
            track_line = gpd.GeoDataFrame(track_line, geometry='geometry', crs=4326)
            track_gdfs = pd.concat([track_gdfs, track_line], axis=0)

    storm_tracks.append(track_gdfs)


nrow = 3
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

minx, miny, maxx, maxy = car_bbox.total_bounds#[-85, 31.5, -75, 37]
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 6),
    tight_layout=True,
    layout='constrained',
    sharex=True, sharey=True)
axs = axs.flatten()
for i in range(len(axs)):
    ax = axs[i]
    if i in first_row:
        tracks = storm_tracks[0]
    elif i in last_row:
        tracks = storm_tracks[2]
    else:
        tracks = storm_tracks[1]

    if i in first_in_row:
        if i in last_row:
            t = tracks.iloc[0:6,:]
            ax.set_xlabel(f"Longitude")
            ax.xaxis.set_visible(True)
            ax.ticklabel_format(style='plain', useOffset=False)
        else:
            t = tracks.iloc[0:7, :]
        ax.set_ylabel(f"Latitude")
        ax.yaxis.set_visible(True)
        ax.ticklabel_format(style='plain', useOffset=False)

    if i in last_in_row:
        if i in last_row:
            t = tracks.iloc[-6:,:]
            ax.set_xlabel(f"Longitude")
            ax.xaxis.set_visible(True)
            ax.ticklabel_format(style='plain', useOffset=False)
        else:
            t = tracks.iloc[-7:, :]

    axs[0].set_title('Present')
    axs[1].set_title('Future')
    row_names = ['Florence', 'Floyd','Matthew']
    for kk in range(nrow):
        axs[first_in_row[kk]].text(-0.20, 0.5, row_names[kk],
                                    horizontalalignment='right',
                                    verticalalignment='center',
                                    rotation='vertical',
                                    transform=axs[first_in_row[kk]].transAxes)
        pos0 = axs[last_in_row[kk]].get_position()  # get the original position

    usa.geometry.plot(ax=ax, color='none', edgecolor='grey')
    #car_bbox.plot(color='none', edgecolor='blue', ax=ax, linewidth=2)
    t.plot(color='black', ax=ax)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

plt.subplots_adjust(wspace=0.0, hspace=0.15, top=0.6)
plt.margins(x=0, y=0)
plt.savefig('WRF_storm_tracks_across_carolinas.png')
plt.close()

