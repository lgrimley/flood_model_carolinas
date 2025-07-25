import xarray as xr
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
import shapely
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

# Plot
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
cat = mod.data_catalog
# Plotting details
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

##################################################################################################################
#Load in data catalog and all the masks
res = 200
proj_crs = 32617
elevation_file = rf'.\03_MODEL_RUNS/subgrid/dep_subgrid_{res}m.tif'
water_mask = rf'.\04_MODEL_OUTPUTS\masks/water_mask_sbgRes{res}m.tif'
basin_mask = rf'.\04_MODEL_OUTPUTS\masks/basin_mask_sbgRes{res}m.tif'

# Load the basins geodataframe
basins = cat.get_geodataframe(data_like=rf'.\04_MODEL_OUTPUTS\masks\basins_shp\huc6_basins.shp')
clip_geom = basins.to_crs(epsg=proj_crs)

# Chunk size specified for dask
chunks_size = {'x': 5000, 'y': 5000}

# Load the elevation dataarray
#elevation_da = cat.get_rasterdataset(elevation_file, geom=clip_geom, chunks=chunks_size)
elevation_da = mod.grid['dep']
elevation_da.name = 'gnd_elevation'

# Load the water body mask dataarray
wb_mask = cat.get_rasterdataset(water_mask, chunks=chunks_size, geom=clip_geom)
wb_mask.rio.write_crs(proj_crs, inplace=True)
wb_mask.name = 'wb_mask'

# Load the basin mask dataarray
basin_mask = cat.get_rasterdataset(basin_mask, crs=proj_crs, chunks=chunks_size, geom=clip_geom)
basin_mask.rio.write_crs(proj_crs, inplace=True)
basin_mask.name = 'basin_mask'
##################################################################################################################
hmin= 0.05
zsmax_ds = xr.open_dataset(r'.\04_MODEL_OUTPUTS\slr_runs\canesm_ssp585_SRL112cm_v2\zsmax_canesm_slr.nc',
                           chunks=chunks_size)
attr_ds = xr.open_dataset(r'.\04_MODEL_OUTPUTS\slr_runs\canesm_ssp585_SRL112cm_v2\attribution_canesm_slr_0.05m.nc',
                          chunks=chunks_size)

# Selected run, mask out data beyond the shapefile
zsmax_da = zsmax_ds.sel(scenario='compound')
attr_da = attr_ds['zsmax_attr']

# Clean up the data array before regridded
zsmax_da = zsmax_da.to_array().squeeze()
zsmax_da.name = 'zsmax'
attr_da.name = 'attr'
zsmax_da = zsmax_da.drop_vars(['variable','scenario'])
attr_da = attr_da.drop_vars(['scenario'])

zsmax_da.rio.write_crs(proj_crs, inplace=True)
attr_da.rio.write_crs(proj_crs, inplace=True)

# Mask out the water body grid cells (wb cell == 1)
mask = (wb_mask != 1)
zsmax_masked = zsmax_da.where(mask)

# Mask out water levels below the ground elevation
mask = (zsmax_masked > elevation_da)
zsmax_masked = zsmax_masked.where(mask)
zsmax_masked.rio.write_crs(proj_crs, inplace=True)

# Calculate the depth above the ground
# Mask out depths smaller than the selected threshold
# Mask out the really large depths -- quarries or from model edge along the coastline
hmax = (zsmax_masked - elevation_da)
hmax.rio.write_crs(proj_crs, inplace=True)
mask = (hmax > hmin) & (hmax <= 10)
hmax_masked = hmax.where(mask)
hmax_masked.rio.write_crs(proj_crs, inplace=True)

# Mask out the attribution code data array
attr_masked = attr_da.where(mask).astype(dtype='int8')
attr_masked.rio.write_crs(proj_crs, inplace=True)

##################################################################################################################
# PLOTTING COMPOUND FREQUENCY
#cmpd_extent_ds = xr.where(cond=(attr_masked == 2) | (attr_masked == 4), x=1, y=0)
cmpd_extent_ds = xr.where(cond=(attr_masked == 2), x=1, y=0)
cmpd_extent_ds.rio.write_crs(proj_crs, inplace=True)

# Split the SLR and NO SLR IDs
tc_ids = cmpd_extent_ds.tc_id.values.tolist()
tc_ids_slr = [x for x in tc_ids if 'SLR' in x]
tc_ids_noslr = [x for x in tc_ids if 'SLR' not in x]

# Select the scenario runs and sum the frequency of being compound across all
cmpd_slr = cmpd_extent_ds.sel(tc_id=tc_ids_slr).sum(dim='tc_id')
cmpd_noslr = cmpd_extent_ds.sel(tc_id=tc_ids_noslr).sum(dim='tc_id')

max_num = len(tc_ids_slr)

# Set up a mask for when both SLR and no SLR do not have compound flooding
data_mask = (cmpd_slr != 0) & (cmpd_noslr != 0)

# calculate the difference in the frequency of compound flooding across the events
diff = cmpd_slr - cmpd_noslr
diff2 = diff.where(data_mask)
#diff = xr.where(mask, np.nan, diff)
#diff2.raster.to_raster(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\05_SLR\version2_TCs_127\cmpd_freq_SLR.tif')

# Load layers - run once because it takes a while...
coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(mod.crs)
coastal_wb_clip = coastal_wb.clip(mod.region)

major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
major_rivers = major_rivers.to_crs(mod.crs)
major_rivers_clip = major_rivers.clip(mod.region)

nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(mod.crs)
nc_major_rivers_clip = nc_major_rivers.clip(mod.region)

##################################################################################################################
# PLOT
perc = diff2/max_num

plot_full_domain = True
if plot_full_domain is True:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), subplot_kw={'projection': utm}, tight_layout=True)
    cmap = plt.cm.get_cmap('BrBG', 5)
    c = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.ListedColormap(c)
    bounds = [-0.5, -0.25, -0.01, 0.01, 0.25, 0.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

    cs = perc.plot(ax=ax,
                   cmap=cmap,
                   norm=norm,
                   extend='neither',
                   shading='auto',
                   add_colorbar=False,
                   zorder=2, alpha=1)
    ax.set_title('')
    ax.set_aspect('equal')
    ax.set_axis_off()
    mod.region.plot(ax=ax, color='lightgrey', edgecolor='none', linewidth=0, zorder=0, alpha=0.7)
    major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    coastal_wb_clip.plot(ax=ax, color='lightblue', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    # major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    # nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    # coastal_wb_clip.plot(ax=ax, color='white', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)
    #clip_geom.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)

    #ax.set_title(f'Frequency of Compound Flooding for Storms RP > 80-yr (hmin:{hmin}m)')
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.075, pos0.y0 + pos0.height * 0.1, 0.035, pos0.y0 + pos0.height * 0.8])
    cbar = fig.colorbar(cs,
                         cax=cax,
                         orientation='vertical',
                         ticks=[-0.5, -0.25, 0, 0.25, 0.5],
                         #label='% Compound Frequency'
                         )
    cbar.ax.set_yticklabels(labels=['50% More Likely\n w/out SLR', '25%', 'Equal Likelihood', '25%', '50% More Likely\nw/ SLR'])

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(rf'.\05_ANALYSIS\05_SLR\coastal_compound_freq_SLR_map.jpg',
                 bbox_inches='tight', dpi=300)
    plt.close()

plot_zooms = True
if plot_zooms is True:
    zoom_axes = [
    [631553.4454,3661553.4271,726419.7563,3750231.4836],
    [727405.6894,3740886.1037,816528.1671,3823974.1323],
    [809790.9378, 3813936.6586, 935189.4691, 3963545.2488]
    ]
    nrow = 1
    ncol = 1
    for i in range(len(zoom_axes)):
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
        import numpy as np
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 6), subplot_kw={'projection': utm}, tight_layout=True)
        cmap = plt.cm.get_cmap('BrBG', 5)
        c = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.ListedColormap(c)
        #c[2] = (0.5, 0.5, 0.5, 1.0)  # RGBA grey
        #colors = ['darkgrey', 'seagreen']
        #cmap = LinearSegmentedColormap.from_list("custom_grey_green", colors)

        # Optional: Discretize the colormap into 5 steps
        cmap = mpl.colors.ListedColormap([cmap(i) for i in np.linspace(0, 1, 5)])
        bounds = [-0.5, -0.25, -0.01, 0.01, 0.25, 0.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

        extent = zoom_axes[i]
        polygon = shapely.geometry.box(*extent)
        cs = perc.plot(ax=ax,
                       cmap=cmap,
                       norm=norm,
                       extend='neither',
                       shading='auto',
                       add_colorbar=False,
                       zorder=3, alpha=1)
        ax.set_title('')
        ax.set_aspect('equal')
        ax.set_axis_off()
        mod.region.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=1, zorder=0, alpha=0.7)
        major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
        nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
        coastal_wb_clip.plot(ax=ax, color='lightblue', edgecolor='darkgrey', linewidth=0.35, zorder=0, alpha=1)
        #mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)
        # clip_geom.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)
        ax.plot(*polygon.exterior.xy, color='black', linewidth=1.5, zorder=3, alpha=1)

        minx, miny, maxx, maxy = extent
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # ax.set_title(f'Frequency of Compound Flooding for Storms RP > 80-yr (hmin:{hmin}m)')
        pos0 = ax.get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0 + pos0.height * 0.1, 0.035, pos0.y0 + pos0.height * 0.8])
        cbar = fig.colorbar(cs,
                            cax=cax,
                            orientation='vertical',
                            ticks=[-0.5, -0.25, 0, 0.25, 0.5],
                            # label='% Compound Frequency'
                            )
        cbar.ax.set_yticklabels(
            labels=['50% More Likely\n w/out SLR', '25%', 'Equal Likelihood', '25%', '50% More Likely\nw/ SLR'])

        plt.subplots_adjust(wspace=0.0, hspace=0)
        plt.margins(x=0, y=0)
        plt.savefig(rf'.\05_ANALYSIS\05_SLR\coastal_compound_freq_SLR_map_zoom{i}.jpg',
                    bbox_inches='tight', dpi=300)
        plt.close()