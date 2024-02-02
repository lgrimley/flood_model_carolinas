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


def downscale_floodmaps(model_root, scenarios, depfile, output_folder, hmin=0.05, fname_key=None,
                        scen_keys=None, gdf_mask=None, output_nc=None, output_tif=False):
    if scen_keys is None:
        scen_keys = scenarios

    da_list = []
    for scen in scenarios:
        if output_tif is True:
            output_tif = hydromt.data_catalog.join(output_folder,
                                                   "floodmap_" + fname_key + "_" + str(hmin) + "_hmin_" + scen + ".tif")
        else:
            output_tif = None

        # Read max water levels from model
        sfincs_mod = SfincsModel(root=os.path.join(model_root, scen), mode='r', data_libs=yml)
        sfincs_mod.read_results(fn_map=os.path.join(model_root, scen, 'sfincs_map.nc'),
                                fn_his=os.path.join(model_root, scen, 'sfincs_his.nc'))
        zsmax = sfincs_mod.results["zsmax"].max(dim='timemax')

        # Downscale results to get depth
        hmax = utils.downscale_floodmap(
            zsmax=zsmax,
            dep=cat.get_rasterdataset(depfile),
            hmin=hmin,
            gdf_mask=gdf_mask,
            reproj_method='bilinear',
            floodmap_fn=output_tif
        )
        da_list.append(hmax)

    # Combine maximum depth maps for all runs
    da = xr.concat(da_list, dim='run')
    da['run'] = xr.IndexVariable('run', scen_keys)

    if output_nc is True:
        da.to_netcdf(os.path.join(output_folder, ("floodmap_sbg_" + str(hmin) + "_hmin.nc")))

    return da


def compute_waterlevel_difference(da, scen_base, scen_keys=None, output_dir=None, output_tif=False):
    # Computer the difference in water level for compound compared to maximum single driver
    da_single_max = da.sel(run=scen_keys).max('run')
    da1 = (da.sel(run=scen_base) - da_single_max).compute()
    da1.name = 'diff. in waterlevel\ncompound - max. single driver'
    da1.attrs.update(unit='m')
    if output_tif is True:
        da1.rio.to_raster(os.path.join(output_dir, 'peak_depth_compound_minus_single_driver_sbg.tif'))
    return da1


yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter2\sfincs_models')
out_dir = os.path.join(os.getcwd(), '00_analysis', 'floodmaps')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# plot_title = 'Florence Climate 4deg Warmer\nMSL 1.10m'
mod_base = ['flor_ensmean_present',
            'flor_ensmean_future_45cm',
            'flor_ensmean_future_110cm',
            'matt_ensmean_present',
            'matt_ensmean_future_45cm',
            'matt_ensmean_future_110cm',
            ]
fld_area_by_driver = []
for m in mod_base:
    scenarios = [m, (m + '_coastal'), (m + '_runoff')]
    scenarios_keys = ['compound', 'coastal', 'runoff']
    hmin = 0.10

    # Downscale water level to subgrid
    da = downscale_floodmaps(model_root=os.getcwd(),
                             fname_key='grid',
                             scenarios=scenarios,
                             depfile=hydromt.data_catalog.join('flor_ensmean_present', "gis", "dep.tif"),
                             hmin=hmin,
                             scen_keys=scenarios_keys,
                             gdf_mask=None,
                             output_folder=out_dir,
                             output_nc=False,
                             output_tif=False
                             )
    da['run'] = xr.IndexVariable('run', scenarios_keys)
    # Calculate difference in water level
    da1 = compute_waterlevel_difference(da=da,
                                        scen_base='compound',
                                        scen_keys=['coastal', 'runoff'],
                                        output_dir=out_dir,
                                        output_tif=False
                                        )
    compound_mask = da1 > hmin
    coastal_mask = da.sel(run='coastal').fillna(0) > da.sel(run=['runoff']).fillna(0).max('run')
    runoff_mask = da.sel(run='runoff').fillna(0) > da.sel(run=['coastal']).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_c = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
            + xr.where(runoff_mask, x=compound_mask + 3, y=0)
            ).compute()
    da_c.name = None

    # Calculate the frequency of the drivers at the grid/subgrid resolution
    da_cd = da_c.data
    unique, counts = np.unique(da_cd, return_counts=True)
    fld_area_by_driver.append(counts)

fld_cells = pd.DataFrame(fld_area_by_driver, columns=['no_water', 'Coastal', 'Compound-Coastal',
                                                      'Runoff', 'Compound-Runoff'])
fld_cells['tmp'] = fld_cells['Compound-Coastal']
fld_cells.drop(['no_water', 'Compound-Coastal'], axis=1, inplace=True)
fld_cells.columns = ['Coastal', 'Runoff', 'Compound-Runoff', 'Compound-Coastal']

fld_cells = fld_cells.T
fld_cells.columns = mod_base
res = 200
fld_area = fld_cells * (res * res) / (1000 ** 2)
fld_area.to_csv(os.path.join(out_dir, 'fld_area_by_driver.csv'))

# scenarios = ['flor_ensmean_present',
#              'flor_ensmean_future_45cm_runoff',
#              'flor_ensmean_future_110cm_coastal',
#              ]
# scenarios_keys = ['compound', 'coastal', 'runoff']
# fld_inundation_by_driver = []
# for scen in scenarios:
#     mod = SfincsModel(root=os.path.join(scen), mode='r', data_libs=yml)
#     mod.read_results()
#
#     tmax = mod.results['tmax'].sum(dim='timemax')
#     tmax_filled = tmax.fillna(0)
#     tmax_filled_float = tmax_filled.astype(np.float64)
#     tmax_hrs = tmax_filled_float * (2.7778**-13)
#     tmax_hrs.rio.to_raster(os.path.join(out_dir, 'test.tif'))
#
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
#     tmax_hrs.plot(ax=ax)
#     plt.savefig(os.path.join(out_dir, 'test.png'),
#                 bbox_inches='tight',
#                 dpi=225)
#     plt.close()
#
#     break
    # da['run'] = xr.IndexVariable('run', scenarios_keys)

''''''
font = {'family': 'Arial',
        'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

# Creating autocpt arguments
# def func(pct, allvalues):
#     absolute = int(pct / 100. * np.sum(allvalues))
#     return "{:.1f}%".format(pct, absolute)
# data = fld_area[mod_base[0]]
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
# wedges, texts, autotexts = ax.pie(data,
#                                   autopct=lambda pct: func(pct, data),
#                                   # labels=pp.index,
#                                   # shadow=True,
#                                   colors=colors,
#                                   explode=(0, 0, 0.1, 0.1),
#                                   startangle=90)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 1.75, box.height])
# ax.legend(wedges,
#           data.index,
#           loc="upper left",
#           bbox_to_anchor=(-0.75, 0.75))
# plt.savefig(os.path.join(out_dir, 'pie.png'), tight_layout=True)
# plt.close()

############
colors = ('navajowhite', 'plum', 'purple', 'orange')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 3),
                        sharex=False, sharey=True)

data = fld_area.iloc[:, 0:3].T
data.plot.bar(ax=axs[0], stacked=True, legend=False, color=colors, title='Florence')
axs[0].set_ylabel('Flooded Area (sq.km.)')
axs[0].set_xticklabels(['Present\nMSL 0.13m', 'Future\nMSL 0.45m', 'Future\nMSL 1.10m'],
                       color='black', rotation=45, horizontalalignment='right')

data = fld_area.iloc[:, 3:7].T
data.plot.bar(ax=axs[1], stacked=True, legend=True, color=colors, title='Matthew')
axs[1].set_xticklabels(['Present\nMSL 0.13m', 'Future\nMSL 0.45m', 'Future\nMSL 1.10m'],
                       color='black', rotation=45, horizontalalignment='right')
axs[0].set_axisbelow(True)
axs[0].grid(axis='y', color='grey', zorder=0, linestyle='--', alpha=0.8)
axs[1].set_axisbelow(True)
axs[1].grid(axis='y', color='grey', zorder=0, linestyle='--', alpha=0.8)
legend_kwargs0 = dict(
    bbox_to_anchor=(1, 0.75),
    title=None,
    loc="upper left",
    frameon=True,
    prop=dict(size=10),
)
axs[1].legend(**legend_kwargs0)
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, 'flooded_area_by_driver_barchart.png'),
            bbox_inches='tight',
            dpi=225)
plt.close()

# Plotting the data on a map with contextual layers
load_lyrs = False
if load_lyrs is True:
    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee', 'Upper Pee Dee'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    l_gdf.set_index('Name', inplace=True)
    basins = l_gdf

    l_gdf = cat.get_geodataframe('carolinas_major_rivers')
    l_gdf.to_crs(epsg=32617, inplace=True)
    major_rivers = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp')
    l_gdf.to_crs(epsg=32617, inplace=True)
    lpd_riv = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(
        ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    cities = l_gdf  # .clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    reservoirs = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
    l_gdf = l_gdf[l_gdf['NAME'].isin(['South Carolina', 'North Carolina'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    l_gdf.set_index('NAME', inplace=True)
    states = l_gdf

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
    l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    roads = l_gdf.clip(states.total_bounds)

    l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
    l_gdf.to_crs(epsg=32617, inplace=True)
    coastal_wb = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(
        ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    cities = l_gdf

    flor_lat = pd.read_table(r'Z:\users\lelise\projects\Carolinas\Chapter2\wrf_output_20231006\ensmean_txt_files'
                             r'\ensmean_minlat.txt',
                             header=None)
    flor_lon = pd.read_table(r'Z:\users\lelise\projects\Carolinas\Chapter2\wrf_output_20231006\ensmean_txt_files'
                             r'\ensmean_minlon.txt',
                             header=None)
    flor_track = pd.DataFrame()
    flor_track['lon'] = flor_lon
    flor_track['lat'] = flor_lat

mod = SfincsModel(scenarios[0], mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

plot_grouped = False
if plot_grouped is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(4.5, 6),
        subplot_kw={'projection': utm},
        tight_layout=True,
        sharex=True,
        sharey=False
    )
    for i in range(len(axs)):
        if i == 0:
            # Plot difference in water level raster
            ckwargs = dict(cmap='seismic', vmin=-0.4, vmax=0.4)
            cs = da1.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
            # Add colorbar
            label = 'Water Level Difference (m)\ncompound - max. individual'
            pos0 = axs[i].get_position()  # get the original position
            cax = fig.add_axes([pos0.x1 - 0.0, pos0.y0 + pos0.height * 0.3, 0.025, pos0.height * 1])
            cbar = fig.colorbar(cs,
                                cax=cax,
                                orientation='vertical',
                                label=label,
                                extend='both',
                                aspect=15,
                                )
            axs[i].set_title('')
            axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(False)
            axs[i].ticklabel_format(style='sci', useOffset=False)
            axs[i].set_aspect('equal')

            basins.plot(ax=axs[i], color='white', edgecolor='black',
                        linewidth=0.1, linestyle='-', zorder=0, alpha=0.25, hatch='xxx')
            mod.region.plot(ax=axs[i], color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)

        if i == 1:
            levels = np.arange(1, 8)
            colors = np.array([
                [252, 141, 98],
                [217, 95, 2],
                [141, 160, 203],
                [117, 112, 179],
                [102, 194, 165],
                [27, 158, 119],
            ]) / 255
            colors = np.hstack([colors, np.ones((6, 1))])
            colors[[0, 2, 4], -1] = 0.7
            cmap, norm = mpl.colors.from_levels_and_colors(levels, colors)
            # Plot the data
            da_c.where(da_c > 0).plot(ax=axs[i], cmap=cmap, norm=norm, add_colorbar=False, zorder=2)
            # Add colorbar
            pos1 = axs[i].get_position()  # get the original position
            cbar_ax = fig.add_axes([pos1.x1 + 0.1, pos1.y0 + pos1.height * 0.3, 0.05, pos1.height * 0.5])
            cm = np.arange(1, 5).reshape((2, 2))
            cbar_ax.imshow(cm, cmap=cmap, norm=norm, aspect='auto')
            cbar_ax.yaxis.tick_right()
            cbar_ax.set_yticks([0, 1])
            cbar_ax.set_yticklabels(['Coastal\n(C+W)', 'Runoff\n(Q+P)'], va='center', rotation=90, fontsize=10)
            cbar_ax.set_xticks([0, 1])
            cbar_ax.set_xticklabels(['Individual', 'Compound'], ha='center', rotation=60, fontsize=10)
            # Fix titles and axis labels
            axs[i].set_title('')
            axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)
            axs[i].ticklabel_format(style='sci', useOffset=False)
            axs[i].set_aspect('equal')

            basins.plot(ax=axs[i], color='lightgrey', edgecolor='black',
                        linewidth=0.1, linestyle='-', zorder=0, alpha=0.25, hatch='xxx')
            mod.region.plot(ax=axs[i], color='white', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)

        # Plot background/geography layers
        basins.plot(ax=axs[i], color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=3, alpha=1)
        roads.plot(ax=axs[i], color='black', edgecolor='none', linewidth=1.15, linestyle='--', zorder=3, alpha=1)
        # reservoirs.plot(ax=axs[i], marker='^', markersize=5, color="black", label='none', zorder=4)
        cities.plot(ax=axs[i], marker='o', markersize=12, color="black",
                    label='none', zorder=4, edgecolor='white', linewidth=0.5)

        # Plotting kwargs
        for label, grow in cities.iterrows():
            x, y = grow.geometry.x, grow.geometry.y
            if label == 'Raleigh':
                ann_kwargs = dict(
                    xytext=(-15, 5),
                    textcoords="offset points",
                    zorder=4,
                    path_effects=[
                        patheffects.Stroke(linewidth=2, foreground="w"),
                        patheffects.Normal(), ], )
            else:
                ann_kwargs = dict(
                    xytext=(6, -15),
                    textcoords="offset points",
                    zorder=4,
                    path_effects=[
                        patheffects.Stroke(linewidth=2, foreground="w"),
                        patheffects.Normal(), ], )
            axs[i].annotate(f'{label}', xy=(x, y), fontsize=10, **ann_kwargs)

        # Setup figure extents
        minx, miny, maxx, maxy = basins.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
    axs[0].set_title(plot_title, loc='center')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, (scenarios[0] + '_driver_analysis.png'))
                , dpi=225, bbox_inches="tight")
    plt.close()
