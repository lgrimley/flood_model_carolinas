import os
import xarray as xr
import numpy as np
from hydromt_sfincs import SfincsModel
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.io as sio
import sys

sys.path.append(r'/')
from src.utils import track_points_to_linestring

# Load in the data catalogs needed for building the model
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
sfincs_mod = SfincsModel(root=r'.\03_MODEL\sfincs_base_mod', mode='r', data_libs=yml_base)
region = sfincs_mod.region
dem = sfincs_mod.grid['dep']

# Load CRS stuff for plotting
wkt = dem.raster.crs.to_wkt()
utm_zone = dem.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

# Load the TC Track file
fname = r'.\02_DATA\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')
tc_tracks_polyline = track_points_to_linestring(tc_tracks, output_shpfile=None)

# Change the directory to the model results
results_dir = r'.\04_RESULTS\ncep'
runs_dir = r'.\03_MODEL\ncep_runs\completed_runs'
output_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep\figs\maps'
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

# Lazily load the zsmax data
file_paths = [os.path.join(results_dir, file) for file in os.listdir(results_dir) if ('attribution' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataset(file, chunks={'x': 300, 'y': 300, 'tc_id': 25})for file in file_paths]
attr_ds = xr.concat(ds_list, dim='tc_id')

# Lazily load the zsmax data
file_paths = [os.path.join(results_dir, file) for file in os.listdir(results_dir) if ('zsmax' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, chunks={'x': 300, 'y': 300, 'tc_id': 25})for file in file_paths]
zsmax_da = xr.concat(da_list, dim='tc_id')
zsmax_ds = zsmax_da.to_dataset(dim='scenario')

tc_ids = zsmax_ds.tc_id.values

tc_ids = [  21,   84,  118,  124,  346,  642,  684,  866,  871,  982, 1149,
       1179, 1425, 1483, 1485, 1821, 2294, 2412, 2567, 2716, 2793, 2832,
       2950, 3143, 3192, 3317, 3740, 3796, 3875, 3989, 4321, 4339, 4442,
       4740, 4885, 4939, 4955, 4983, 5014, 5015]
for tc_id in tc_ids:
    try:
        # Get storm track
        tc_track_gdf = tc_tracks_polyline[tc_tracks_polyline['tc_id'] == tc_id]
        tc_track_gdf = tc_track_gdf.to_crs(32617)

        # Load precip boundary condition
        precip_filepath = os.path.join(runs_dir, f'TC_{str(tc_id).zfill(4)}', 'sfincs_bc_inputs', 'precip_2d.nc')
        precip = sfincs_mod.data_catalog.get_rasterdataset(precip_filepath)
        zsmax = zsmax_ds.sel(tc_id=tc_id)['compound']
        hmax = zsmax - dem
        hmax = hmax.where(hmax > 0.05)

        # Plot
        font = {'family': 'Arial', 'size': 10}
        mpl.rc('font', **font)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 7), subplot_kw={'projection': utm},
                                tight_layout=True, sharex=True, sharey=False)
        # Map total precipitation
        ax=axs[0]
        ckwargs = dict(vmin=0, vmax=400, cmap='jet')
        cs = precip.sum(dim='time').plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
        region.plot(ax=ax, color='none', edgecolor='white', linewidth=1, zorder=1, alpha=1)
        tc_track_gdf.geometry.plot(ax=ax, color='red', linewidth=2, zorder=2)

        # Set figure extents
        minx, miny, maxx, maxy = region.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Add colorbar
        pos0 = ax.get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 + 0.075, pos0.y0 + 0.05, 0.05, pos0.height * 0.8])
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label='Total Precipitation (mm)', extend='max')
        ax.set_axis_off()
        ax.set_title(f'{tc_id}')

        # Plot Peak Flood Depth - Compound
        ax=axs[1]
        ckwargs = dict(cmap='Blues', vmin=0.05, vmax=10)
        cs = hmax.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=2)

        # Plot background/geography layers
        region.plot(ax=ax, color='grey', edgecolor='none', linewidth=0.5, zorder=1, alpha=1)
        region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
        tc_track_gdf.geometry.plot(ax=ax, color='black', linewidth=2, zorder=2)

        minx, miny, maxx, maxy = region.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        pos0 = ax.get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 - 0.02, pos0.y0 + -0.05, 0.04, pos0.height * 0.5])
        label = 'Max Water Depth (m)'
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='max',
                            ticks=[0.05, 1, 2, 4, 6, 8, 10])
        ax.set_title(f'{tc_id}')
        ax.set_axis_off()
        plt.margins(x=0, y=0)
        plt.savefig(os.path.join(output_dir, f'{tc_id}_total_precip_and_max_depth.png'), dpi=225, bbox_inches="tight")
        plt.close()

        ''' Plot 2: Peak Flood Extent Attributed '''
        attr = attr_ds.sel(tc_id=tc_id)
        da_diff = attr['zsmax_diff']
        da_c = attr['zsmax_attr']
        da_c = da_c.where(da_c > 0)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 7), subplot_kw={'projection': utm},
                                tight_layout=True, sharex=True, sharey=False)
        for i in range(len(axs)):
            if i == 1:
                # Plot difference in water level raster
                ckwargs = dict(cmap='seismic', vmin=-0.5, vmax=0.5)
                cs = da_diff.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)

                # Add colorbar
                label = 'Water Level Difference (m)\ncompound - max. individual'
                pos0 = axs[i].get_position()  # get the original position
                cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0 + 0.02, 0.05, pos0.height * 0.8])
                cbar = fig.colorbar(cs,
                                    cax=cax,
                                    orientation='vertical',
                                    label=label,
                                    extend='both')
                axs[i].set_title('')
                axs[i].set_ylabel(f"y coord UTM zone {utm_zone} (m)")
                axs[i].yaxis.set_visible(True)
                axs[i].set_xlabel(f"x coord UTM zone {utm_zone} (m)")
                axs[i].xaxis.set_visible(True)
                axs[i].ticklabel_format(style='sci', useOffset=False)
                axs[i].set_aspect('equal')

                region.plot(ax=axs[i], color='grey', edgecolor='none', linewidth=0.5, zorder=1, alpha=1)
                region.plot(ax=axs[i], color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
                tc_track_gdf.geometry.plot(ax=axs[i], color='black', linewidth=2, zorder=2)

            if i == 0:
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
                da_c.plot(ax=axs[i], cmap=cmap, norm=norm, add_colorbar=False, zorder=2)

                # Add colorbar
                pos1 = axs[i].get_position()  # get the original position
                cbar_ax = fig.add_axes([pos1.x1 + 0.05, pos1.y0 + pos1.height * 0.4, 0.08, pos1.height])
                cm = np.arange(1, 5).reshape((2, 2))
                cbar_ax.imshow(cm, cmap=cmap, norm=norm, aspect='auto')
                cbar_ax.yaxis.tick_right()
                cbar_ax.set_yticks([0, 1])
                cbar_ax.set_yticklabels(['Coastal\n', 'Runoff\n'], va='center', rotation=90, fontsize=10)
                cbar_ax.set_xticks([0, 1])
                cbar_ax.set_xticklabels(['Individual', 'Compound'], ha='center', rotation=60, fontsize=10)

                # Fix titles and axis labels
                axs[i].set_title('')
                axs[i].set_ylabel(f"y coord UTM zone {utm_zone} (m)")
                axs[i].yaxis.set_visible(True)
                axs[i].set_xlabel(f"x coord UTM zone {utm_zone} (m)")
                axs[i].xaxis.set_visible(False)
                axs[i].ticklabel_format(style='sci', useOffset=False)
                axs[i].set_aspect('equal')
                axs[i].set_title(f'{tc_id}')

                region.plot(ax=axs[i], color='white', edgecolor='none', linewidth=0.5, zorder=1, alpha=1)
                region.plot(ax=axs[i], color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
                tc_track_gdf.geometry.plot(ax=axs[i], color='black', linewidth=2, zorder=2)

            # Setup figure extents
            minx, miny, maxx, maxy = region.total_bounds
            axs[i].set_xlim(minx, maxx)
            axs[i].set_ylim(miny, maxy)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(x=0, y=0)
        plt.savefig(os.path.join(output_dir, f'{tc_id}_zsmax_attribution.png'), dpi=225,
                    bbox_inches="tight")
        plt.close()

        print(tc_id)
    except:
        print(f'issue with {tc_id}')

