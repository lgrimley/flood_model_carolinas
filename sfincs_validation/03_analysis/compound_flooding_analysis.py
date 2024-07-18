import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
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


font = {'family': 'Arial',
        'size': 10
        }
mpl.rc('font', **font)

# Filepath to data catalog yml
yml = os.path.join(r'Z:\users\lelise\data\data_catalog.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(
    r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2016_Matthew\matt_hindcast_v6_200m_LPD2m_avgN')
model_root = os.getcwd()

# Create directories
out_dir = os.path.join(model_root, 'scenarios', '00_driver_analysis')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scenarios = ['compound']

# Downscale water level to subgrid
da = downscale_floodmaps(model_root=os.path.join(model_root, 'scenarios'),
                         fname_key='sbg',
                         scenarios=scenarios,
                         #depfile=hydromt.data_catalog.join(model_root, "subgrid", "dep_subgrid.tif"),
                         depfile=hydromt.data_catalog.join(model_root, "gis", "dep.tif"),
                         hmin=0.05,
                         scen_keys=['compound'],
                         gdf_mask=None,
                         output_folder=out_dir,
                         output_nc=False,
                         output_tif=True
                         )

# Load model
mod = SfincsModel(root=model_root, mode='r', data_libs=yml)

'''Load contextual layers for plotting'''
load_lyrs = True
if load_lyrs is True:
    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    basins = l_gdf

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
    l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    roads = l_gdf.clip(basins.total_bounds)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    cities = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    reservoirs = l_gdf.clip(basins)

    tc_tracks = cat.get_geodataframe(r'Z:\users\lelise\geospatial\tropical_cyclone\hurricane_tracks\IBTrACS.NA.list'
                                     r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
    tc_tracks.to_crs(epsg=32617, inplace=True)
    tc_tracks = tc_tracks.clip(basins.total_bounds)

'''Peak Flood Depth - Compound'''
plot_flddpth = True
if plot_flddpth is True:
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4.5))

    # Plot difference in water level raster
    ckwargs = dict(cmap='Blues', vmin=0, vmax=10)
    cs = da.sel(run='compound').plot(ax=ax, add_colorbar=False, **ckwargs, zorder=2)

    # Plot background/geography layers
    basins.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, linestyle='-', zorder=0, alpha=0.25,
                hatch='xxx')
    mod.region.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
    basins.plot(ax=ax, color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=3, alpha=1)
    roads.plot(ax=ax, color='black', edgecolor='none', linewidth=1, linestyle='--', zorder=3, alpha=1)
    reservoirs.plot(ax=ax, marker='^', markersize=5, color="black", label='none', zorder=4)
    cities.plot(ax=ax, marker='o', markersize=12, color="black", label='none', zorder=4, edgecolor='white',
                linewidth=0.5)

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
        ax.annotate(f'{label}', xy=(x, y), fontsize=10, **ann_kwargs)

    minx, miny, maxx, maxy = basins.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.01, pos0.y0 + -0.05, 0.015, pos0.height * 0.5])
    label = 'Max Water Depth\n(m+NAVD88)'
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='max')

    ax.set_title('')
    ax.set_axis_off()
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'compound_max_depth.png'), dpi=225, bbox_inches="tight")
    plt.close()

'''Grouped Forcings'''
# Difference in Peak Water Level - Runoff + Coastal Drivers
da1 = compute_waterlevel_difference(da=da,
                                    scen_base='compound',
                                    scen_keys=['coastal', 'runoff'],
                                    output_dir=out_dir,
                                    output_tif=True)

# Select areas where the difference in water level between the compound
# and max single driver is greater than a threshold (dh)
# Create mask for single driver flooding and assign number based on driver of flooding for plotting
dh = 0.05
compound_mask = da1 > dh
coastal_mask = da.sel(run='coastal').fillna(0) > da.sel(run=['runoff']).fillna(0).max('run')
runoff_mask = da.sel(run='runoff').fillna(0) > da.sel(run=['coastal']).fillna(0).max('run')
# runoff_mask = np.logical_and(runoff_mask, da1 >= 0)
assert ~np.logical_and(runoff_mask, coastal_mask).any()  # and ~np.logical_and(coastal_mask, runoff_mask).any()
da_c = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
        + xr.where(runoff_mask, x=compound_mask + 3, y=0)
        ).compute()
da_c.name = None

# Load CRS stuff for plotting
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(basins.buffer(10000).total_bounds)[[0, 2, 1, 3]]

plot_grouped = True
if plot_grouped is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(6.5, 6.5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        sharex=True,
        sharey=False
    )
    for i in range(len(axs)):
        if i == 0:
            # Plot difference in water level raster
            ckwargs = dict(cmap='seismic', vmin=-0.2, vmax=0.2)
            cs = da1.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
            # Add colorbar
            label = 'Water Level Difference (m)\ncompound - max. individual'
            pos0 = axs[i].get_position()  # get the original position
            cax = fig.add_axes([pos0.x1 - 0.1, pos0.y0 + + pos0.height * 0.3, 0.025, pos0.height * 1])
            cbar = fig.colorbar(cs,
                                cax=cax,
                                orientation='vertical',
                                label=label,
                                extend='both')
            axs[i].set_title('')
            axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(False)
            axs[i].ticklabel_format(style='plain', useOffset=False)
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
            cbar_ax = fig.add_axes([pos1.x1 + 0.05, pos1.y0 + pos1.height * 0.3, 0.05, pos1.height * 0.5])
            cm = np.arange(1, 5).reshape((2, 2))
            cbar_ax.imshow(cm, cmap=cmap, norm=norm, aspect='auto')
            cbar_ax.yaxis.tick_right()
            cbar_ax.set_yticks([0, 1])
            cbar_ax.set_yticklabels(['Coastal\n(C+W)', 'Runoff\n(Q+P)'], va='center', rotation=90, fontsize=10)
            cbar_ax.set_xticks([0, 1])
            cbar_ax.set_xticklabels(['Individual', 'Compound'], ha='center', rotation=60, fontsize=10)
            # Fix titles and axis labels
            axs[i].set_title('')
            axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)
            axs[i].ticklabel_format(style='plain', useOffset=False)
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
        tc_tracks[tc_tracks['NAME'] == 'FLORENCE'].plot(ax=axs[i], color='slategray', edgecolor='none',
                                                        linewidth=2, linestyle='-', zorder=3, alpha=0.80)

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

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'runoff_coastal_compound_driver_analysis.png')
                , dpi=225, bbox_inches="tight")
    plt.close()

'''All Forcings'''

scenarios = ['compound', 'coastal_wl', 'wind', 'rainfall', 'discharge']

# Downscale water level to subgrid
da = downscale_floodmaps(model_root=os.path.join(model_root, 'scenarios'),
                         fname_key='grd',
                         scenarios=scenarios,
                         # depfile=hydromt.data_catalog.join(model_root, "subgrid", "dep_subgrid.tif"),
                         depfile=hydromt.data_catalog.join(model_root, "gis", "dep.tif"),
                         hmin=0.05,
                         scen_keys=scenarios,
                         gdf_mask=None,
                         output_folder=out_dir,
                         output_nc=True,
                         output_tif=False
                         )

# Load model
mod = SfincsModel(root=model_root, mode='r', data_libs=yml)

da1 = compute_waterlevel_difference(da=da,
                                    scen_base='compound',
                                    scen_keys=['coastal_wl', 'wind', 'discharge', 'rainfall'],
                                    output_dir=out_dir,
                                    output_tif=False
                                    )
dh = 0.05
compound_mask = da1 > dh
surge_mask = da.sel(run='coastal_wl').fillna(0) > da.sel(
    run=['discharge', 'rainfall', 'wind']).fillna(0).max('run')
coastal_mask = da.sel(run='wind').fillna(0) > da.sel(
    run=['discharge', 'rainfall', 'coastal_wl']).fillna(0).max('run')
discharge_mask = da.sel(run='discharge').fillna(0) > da.sel(
    run=['wind', 'rainfall', 'coastal_wl']).fillna(0).max('run')
precip_mask = da.sel(run='rainfall').fillna(0) > da.sel(
    run=['wind', 'discharge', 'coastal_wl']).fillna(0).max('run')
# precip_mask = np.logical_and(precip_mask, da1 >= 0)

assert ~np.logical_and(precip_mask, surge_mask).any() and ~np.logical_and(precip_mask,
                                                                          coastal_mask).any() and ~np.logical_and(
    precip_mask, discharge_mask).any()

assert ~np.logical_and(discharge_mask, surge_mask).any() and ~np.logical_and(discharge_mask,
                                                                             coastal_mask).any()
assert ~np.logical_and(surge_mask, coastal_mask).any()

da_c = (
        + xr.where(surge_mask, x=compound_mask + 1, y=0)
        + xr.where(coastal_mask, x=compound_mask + 3, y=0)
        + xr.where(discharge_mask, x=compound_mask + 5, y=0)
        + xr.where(precip_mask, x=compound_mask + 7, y=0)
).compute()
da_c.name = None

plot_grouped = True
if plot_grouped is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(6.5, 6.5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        sharex=True,
        sharey=False
    )
    for i in range(len(axs)):
        if i == 0:
            # Plot difference in water level raster
            ckwargs = dict(cmap='seismic', vmin=-0.2, vmax=0.2)
            cs = da1.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
            # Add colorbar
            label = 'Water Level Difference (m)\ncompound - max. individual'
            pos0 = axs[i].get_position()  # get the original position
            cax = fig.add_axes([pos0.x1 - 0.1, pos0.y0 + + pos0.height * 0.3, 0.025, pos0.height * 1])
            cbar = fig.colorbar(cs,
                                cax=cax,
                                orientation='vertical',
                                label=label,
                                extend='both')
            axs[i].set_title('')
            axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(False)
            axs[i].ticklabel_format(style='plain', useOffset=False)
            axs[i].set_aspect('equal')

            basins.plot(ax=axs[i], color='white', edgecolor='black',
                        linewidth=0.1, linestyle='-', zorder=0, alpha=0.25, hatch='xxx')
            mod.region.plot(ax=axs[i], color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)

        if i == 1:
            levels = np.arange(1, 12)
            # https://htmlcolorcodes.com/
            colors = np.array([
                [251, 212, 115],  # Yellow
                [251, 211, 52],
                [252, 141, 98],  # Orange
                [217, 95, 2],
                # [102, 194, 165],  # Green
                # [27, 158, 119],
                [23, 190, 207],  # Blue
                [35, 167, 183],
                [234, 108, 238],  # Pink
                [196, 60, 200],
                # [71, 108, 236],  # Dark Blue
                # [28, 56, 149],
                [141, 160, 203],  # Purple
                [117, 112, 179],
            ]) / 255
            colors = np.hstack([colors, np.ones((10, 1))])
            colors[[0, 2, 4, 6, 8], -1] = 0.7
            cmap, norm = mpl.colors.from_levels_and_colors(levels, colors)
            # Plot the data
            da_c.where(da_c > 0).plot(ax=axs[i], cmap=cmap, norm=norm, add_colorbar=False, zorder=2)
            # Add colorbar
            pos1 = axs[i].get_position()  # get the original position
            cbar_ax = fig.add_axes([pos1.x1 + 0.065, pos1.y0 + pos1.height * 0.3, 0.05, pos1.height * 0.5])
            cm = np.arange(1, 9).reshape((4, 2))
            cbar_ax.imshow(cm, cmap=cmap, norm=norm, aspect='auto')
            cbar_ax.yaxis.tick_right()
            cbar_ax.set_yticks([0, 1, 2, 3])
            # cbar_ax.set_yticklabels(['surge', 'discharge', 'rainfall'], va='center', rotation=30)
            cbar_ax.set_yticklabels(['C', 'W', 'Q', 'P'], va='center', rotation=90)
            cbar_ax.set_xticks([0, 1])
            cbar_ax.set_xticklabels(['Individual', 'Compound'], ha='center', rotation=60)
            # Fix titles and axis labels
            axs[i].set_title('')
            axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
            axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)
            axs[i].ticklabel_format(style='plain', useOffset=False)
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
        tc_tracks[tc_tracks['NAME'] == 'FLORENCE'].plot(ax=axs[i], color='slategray', edgecolor='none',
                                                        linewidth=2, linestyle='-', zorder=3, alpha=0.80)

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

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'compound_analysis_sfincs_byforcing.png')
                , dpi=225, bbox_inches="tight")
    plt.close()
