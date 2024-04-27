import os
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import colors, patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as ss
from scipy.optimize import curve_fit


# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley


def calc_windspd(da):
    wind_u = da['wind_u']
    wind_v = da['wind_v']
    wind_spd = ((wind_u ** 2) + (wind_v ** 2)) ** 0.5
    da['wind_spd'] = wind_spd
    return da


def subset_data_to_calc_stats(data, min_threshold, max_threshold, variable):
    # Subset data
    data = xr.where(cond=(data > min_threshold), x=data, y=np.nan)
    data = xr.where(cond=(data < max_threshold), x=data, y=np.nan)

    # Convert to array
    data = data.to_dataframe()
    data = data.reset_index()
    data = data[variable]
    data.dropna(inplace=True)

    return np.array(data)


def expon_func(x, a):
    return a * np.exp(-a * x)


def fit_expon_dist(data, lower_threshold, upper_threshold, nbins):
    # Fit exponential distribution and save params
    loc, scale = ss.expon.fit(data, floc=0)
    counts, bins = np.histogram(data, bins=nbins, range=[lower_threshold, upper_threshold], density=True)
    bin_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    popt, pcov = curve_fit(expon_func, xdata=bin_centers, ydata=counts)
    return loc, scale, popt, pcov


def empirical_stats_and_curvefit(runs_dict, variable, lower_threshold, upper_threshold, bin_size,
                                 masks=None, bbox=None):
    # Histogram details
    bin_edges = np.arange(lower_threshold, upper_threshold, bin_size)
    nbins = len(bin_edges)

    # Empty dictionary and dataframe to save output to
    histogram_data = {'floyd': {'present': {}, 'future': {}},
                      'matthew': {'present': {}, 'future': {}},
                      'florence': {'present': {}, 'future': {}}
                      }
    histogram_info = pd.DataFrame()

    storms = list(histogram_data.keys())
    climates = list(histogram_data[storms[0]].keys())

    counter = 0
    for storm in storms:
        print(storm)

        for climate in climates:
            print(climate)

            runs = runs_dict[storm][climate].keys()
            for run in runs:
                print(run)

                # Load in the data for the subplot and fit distribution
                run_data = runs_dict[storm][climate][run][variable]

                # Clip/mask data
                if masks is not None:
                    run_data = run_data.where(masks[counter])

                if bbox is not None:
                    minx, miny, maxx, maxy = bbox
                    run_data = run_data.sel(x=slice(minx, maxx), y=slice(miny, maxy))

                run_data.max(dim='time').raster.to_raster(f'{storm}_{climate}_{run}_{variable}.tif')

                data = subset_data_to_calc_stats(data=run_data,
                                                 min_threshold=lower_threshold,
                                                 max_threshold=upper_threshold,
                                                 variable=variable)
                histogram_data[storm][climate][f'{run}'] = data

                # Get stat info on the empirical data
                df1 = pd.DataFrame(data)
                df = df1.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
                df = df.T

                # loc, scale, popt, pcov = fit_expon_dist(data=np.array(data),
                #                                         lower_threshold=lower_threshold,
                #                                         upper_threshold=lower_threshold,
                #                                         nbins=nbins)
                # df['loc'] = loc
                # df['scale'] = scale
                # df['popt'] = popt[0]
                # df['pcov'] = pcov[0]

                df['climate'] = climate
                df['storm'] = storm
                df['run'] = run
                df['run_id'] = f'{storm}_{climate}_{run}'
                histogram_info = pd.concat([histogram_info, df])

        counter += 1

    histogram_info.set_index('run_id', inplace=True, drop=True)

    return histogram_data, histogram_info


def plot_data(data_for_plot, data_info_for_plot, lower_threshold, upper_threshold, bin_size, axis_label,
              plot_hist_pdf=True, plot_dist_cdf=False, plot_ecdf=True,
              ax_xlim1=None, ax_ylim1=None, figsize1=(8, 12), figsize2=(8, 4)):
    # Histogram details
    bin_edges = np.arange(lower_threshold, upper_threshold, bin_size)
    nbins = len(bin_edges)

    if plot_hist_pdf is True:
        for storm in storms:
            climates = list(runs_dict[storm].keys())
            runs = list(runs_dict[storm][climates[0]].keys())

            # Info for controlling subplots
            nrow, ncol = [len(runs), len(climates)]
            n_subplots = nrow * ncol
            first_in_row = np.arange(0, n_subplots, ncol)
            last_row = np.arange(n_subplots - ncol, n_subplots, 1)

            # Create plot
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize1, tight_layout=True,
                                    layout='constrained', sharex=True, sharey=True)

            for k in range(nrow):
                run = runs[k]
                for i in range(ncol):
                    # Get the climate scenario
                    climate = climates[i]
                    # Get subplot axes
                    ax = axs[k, i]

                    # Load in the data for the subplot and fit distribution
                    data = data_for_plot[storm][climate][run]
                    run_id = f'{storm}_{climate}_{run}'
                    loc = data_info_for_plot.loc[run_id, 'loc']
                    scale = data_info_for_plot.loc[run_id, 'scale']
                    params = (loc, scale)
                    popt = data_info_for_plot.loc[run_id, 'popt']

                    # Plot density histogram and PDF
                    ax.hist(data, bins=nbins, range=[lower_threshold, upper_threshold],
                            density=True, histtype='stepfilled', alpha=0.4)
                    xx = np.linspace(0, upper_threshold, 1000)
                    ax.plot(xx, ss.expon.pdf(xx, *params), lw=2.0, color='green',
                            label=r"$\lambda = $" + str(round(1 / params[1], 3)))
                    ax.plot(xx, expon_func(xx, popt), lw=2.0, color='red',
                            label=r"$\lambda = $" + str(round(popt, 3)))

                    # Add subplot legend, title, etc.
                    ax.set_xlim(ax_xlim1)
                    ax.set_ylim(ax_ylim1)
                    ax.legend(loc='best')
                    ax.set_title(f'{climate} {run}')

            axs = axs.flatten()
            for ii in range(len(axs)):
                axs[ii].yaxis.set_visible(False)
                axs[ii].xaxis.set_visible(False)
                if ii in first_in_row:
                    axs[ii].yaxis.set_visible(True)
                    axs[ii].set_ylabel('Density')
                if ii in last_row:
                    axs[ii].xaxis.set_visible(True)
                    axs[ii].set_xlabel(axis_label)

            # Save and close plot
            plt.subplots_adjust(wspace=0.05, hspace=0.3, top=0.96)
            plt.margins(x=0, y=0)
            plt.suptitle(storm)
            plt.savefig(
                os.path.join(os.getcwd(), f'{storm}_{var}_histograms_{lower_threshold}_to_{upper_threshold}.png'),
                bbox_inches='tight', dpi=255)
            plt.close()

    if plot_dist_cdf is True:
        for storm in storms:
            # Info for controlling subplots
            nrow, ncol = [2, 4]
            n_subplots = nrow * ncol
            first_in_row = np.arange(0, n_subplots, ncol)
            last_row = np.arange(n_subplots - ncol, n_subplots, 1)

            # Create plot
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize2, tight_layout=True,
                                    layout='constrained', sharex=True, sharey=True)
            axs = axs.flatten()
            if storm == 'matthew':
                axs[-1].set_visible(False)
            runs = list(runs_dict[storm]['present'].keys())

            for i in range(len(runs)):
                run = runs[i]
                ax = axs[i]
                # Get exponential dist parameters
                run_id = f'{storm}_present_{run}'
                params1 = (data_info_for_plot.loc[run_id, 'loc'], data_info_for_plot.loc[run_id, 'scale'])

                run_id = f'{storm}_future_{run}'
                params2 = (data_info_for_plot.loc[run_id, 'loc'], data_info_for_plot.loc[run_id, 'scale'])

                # Plot CDF
                xx = np.linspace(0, upper_threshold, 1000)
                ax.plot(xx, ss.expon.cdf(xx, *params1), lw=2.5, color='green', label='Present')
                ax.plot(xx, ss.expon.cdf(xx, *params2), lw=2.5, linestyle='--', color='orange', label='Future',
                        alpha=0.9)

                ax.set_ylim([0, 1])
                ax.set_title(f'{run}')
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.grid(which='minor', linewidth=0.5, alpha=0.5, linestyle='--', color='grey')
                ax.grid(which='major', linewidth=1, alpha=0.8, linestyle='-', color='darkgrey')

                if i in first_in_row:
                    ax.yaxis.set_visible(True)
                    ax.set_ylabel('Percentile')
                if i in last_row:
                    ax.xaxis.set_visible(True)
                    ax.set_xlabel(axis_label)

            axs[-1].legend(loc='best')
            # Save and close plot
            plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.91)
            plt.margins(x=0, y=0)
            plt.suptitle(storm)
            plt.savefig(os.path.join(os.getcwd(), f'{storm}_{var}_cdfs_{lower_threshold}_to_{upper_threshold}.png'),
                        bbox_inches='tight',
                        dpi=255)
            plt.close()

    if plot_ecdf is True:
        for storm in storms:
            # Info for controlling subplots
            nrow, ncol = [2, 4]
            n_subplots = nrow * ncol
            first_in_row = np.arange(0, n_subplots, ncol)
            last_row = np.arange(n_subplots - ncol, n_subplots, 1)

            # Create plot
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize2, tight_layout=True,
                                    layout='constrained', sharex=True, sharey=True)
            axs = axs.flatten()
            if storm == 'matthew':
                axs[-1].set_visible(False)
            runs = list(runs_dict[storm]['present'].keys())

            for i in range(len(runs)):
                run = runs[i]
                ax = axs[i]

                ecdf1 = ss.ecdf(data_for_plot[storm]['present'][run])
                ecdf2 = ss.ecdf(data_for_plot[storm]['future'][run])

                # Plot CDF
                ecdf1.cdf.plot(linewidth=2.5, color='green', label='Present', ax=ax)
                ecdf2.cdf.plot(linewidth=2.5, linestyle='--', color='orange', label='Future', ax=ax)

                ax.set_ylim([0, 1])
                ax.set_title(f'{run}')
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.grid(which='minor', linewidth=0.5, alpha=0.5, linestyle='--', color='grey')
                ax.grid(which='major', linewidth=0.6, alpha=0.5, linestyle='-', color='darkgrey')

                if i in first_in_row:
                    ax.yaxis.set_visible(True)
                    ax.set_ylabel('Percentile')
                if i in last_row:
                    ax.xaxis.set_visible(True)
                    ax.set_xlabel(axis_label)

            axs[-1].legend(loc='best')
            # Save and close plot
            plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.91)
            plt.margins(x=0, y=0)
            plt.suptitle(storm)
            plt.savefig(os.path.join(os.getcwd(), f'{storm}_{var}_ecdfs_{lower_threshold}_to_{upper_threshold}.png'),
                        bbox_inches='tight',
                        dpi=255)
            plt.close()


def calculate_scale_factors(data_stats):
    stat_cols = ['mean', '1%', '10%', '25%', '50%', '75%', '90%', '99%']

    # Create a dataframe to populate with Scale Factors
    scale_factors_df = pd.DataFrame()

    for storm in data_info['storm'].unique():
        runs = runs_dict[storm]['present'].keys()
        for run in runs:
            sf_stats = []
            for x in stat_cols:
                x1 = data_stats.loc[f'{storm}_present_{run}', x]  # Present values
                x2 = data_stats.loc[f'{storm}_future_{run}', x]  # Future values
                sf = (x2 - x1) / x1  # Scale factor
                sf_stats.append(sf)
            sf_stats.append(f'{storm}_{run}')
            scale_factors_df = pd.concat([scale_factors_df, pd.DataFrame(sf_stats).T])

    # Cleanup output dataframe with Scale Factors
    scale_factors_df.columns = stat_cols + ['Run']
    scale_factors_df.set_index('Run', inplace=True, drop=True)
    scale_factors_df = scale_factors_df.astype('float').round(decimals=3)

    return scale_factors_df


'''  Data and plotting info  '''
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

state_boundaries = cat.get_geodataframe(
    r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
state_boundaries.to_crs(epsg=4326, inplace=True)
aoi = state_boundaries[state_boundaries['NAME'].isin(['South Carolina', 'North Carolina',
                                                      'Georgia', 'Virginia', 'Florida'])]
state_boundaries.set_index('NAME', inplace=True)
aoi_model = state_boundaries[state_boundaries.index.isin(['South Carolina', 'North Carolina'])]


'''  Load WRF output and calculate wind speed '''
wd = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\met'
os.chdir(wd)
storms = ['floyd', 'matthew', 'florence']
climates = ['present', 'future']
runs_dict = {'floyd': {'present': {}, 'future': {}},
             'matthew': {'present': {}, 'future': {}},
             'florence': {'present': {}, 'future': {}}
             }
wrf_storm_grids = []
for storm in storms:
    for climate in climates:
        met_dir = f'{climate}_{storm}'
        for file in os.listdir(os.path.join(os.getcwd(), met_dir)):
            if file.endswith('.nc'):
                run_name = file.split('.')[0].split('_')[-1]
                run_filepath = os.path.join(os.getcwd(), met_dir, file)

                # Read in netcdf of WRF output, calculate wind speed
                da = cat.get_rasterdataset(run_filepath)
                da = calc_windspd(da)

                # Add data to dictionary
                runs_dict[storm][climate][f'{run_name}'] = da
                print(run_filepath)
    wrf_storm_grids.append(da)


# Setup masks for each storm grid
wrf_storm_grids_masks = []
state_boundaries['mask'] = 1.0
for wrf_grid in wrf_storm_grids:
    mask_ras = wrf_grid.raster.rasterize(state_boundaries, "mask", nodata=np.nan, all_touched=False)
    mask_ras = xr.where(mask_ras == 1.0, False, True)
    wrf_storm_grids_masks.append(mask_ras)


# User Input
var = 'wind_spd'
bin_size = 5
upper_threshold = 100
thresholds = np.arange(0, 35, bin_size)
overwrite_files = False
plt_label = 'Wind Speed\nLower Threshold\n(m/s)'
axis_label = 'Wind Speed (m/s)'
bbox = None


# Run script!
info_threshold = []
sf_threshold = []
for lower_threshold in thresholds:
    out_dir = os.path.join(wd, 'scale_factors', f'{var}_thresh_{lower_threshold}_to_{upper_threshold}')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
    os.chdir(out_dir)

    data, data_info = empirical_stats_and_curvefit(runs_dict=runs_dict,
                                                   variable=var,
                                                   bbox=bbox,
                                                   masks=wrf_storm_grids_masks,
                                                   lower_threshold=lower_threshold,
                                                   upper_threshold=upper_threshold,
                                                   bin_size=bin_size)
    data_info.to_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_data.csv')

    plot_data(data_for_plot=data, data_info_for_plot=data_info,
              lower_threshold=lower_threshold, upper_threshold=upper_threshold,
              bin_size=bin_size, plot_hist_pdf=False, plot_dist_cdf=False, plot_ecdf=True,
              axis_label=axis_label,
              ax_xlim1=[lower_threshold, 40], ax_ylim1=[0.0, 0.1], figsize1=(8, 12), figsize2=(8, 4))

    scale_factor_df = calculate_scale_factors(data_stats=data_info)
    scale_factor_df.to_csv(f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_scalefactors.csv')

    info_threshold.append(data_info)
    sf_threshold.append(scale_factor_df)

# Plotting scale factors by threshold
os.chdir(os.path.join(wd, 'scale_factors'))
ax_ylim = [-0.15, 0.25]
for storm in list(runs_dict.keys()):
    # Info for controlling subplots
    nrow, ncol = [2, 4]
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_row = np.arange(n_subplots - ncol, n_subplots, 1)
    figsize = (8, 4)

    # Create plot
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, tight_layout=True,
                            layout='constrained', sharex=True, sharey=True)
    axs = axs.flatten()
    if storm == 'matthew':
        axs[-1].set_visible(False)

    runs = list(runs_dict[storm]['present'].keys())
    for i in range(len(runs)):
        run = f'{storm}_{runs[i]}'
        ax = axs[i]

        # Subset data for plotting
        run_df = pd.DataFrame(columns=thresholds)
        for ii in range(len(thresholds)):
            d = sf_threshold[ii].loc[run]
            run_df[thresholds[ii]] = d[d.index.isin(['mean', '5%', '50%', '99%'])]

        run_df.T.plot(ax=ax, legend=False, lw=2,
                      color=['black', 'lightcoral', 'deepskyblue', 'crimson'])

        ax.set_xlim((min(thresholds), max(thresholds)))
        ax.set_ylim(ax_ylim)
        ax.set_title(f'{runs[i]}')

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which='minor', linewidth=0.5, alpha=0.5, linestyle='--', color='grey')
        ax.grid(which='major', linewidth=1, alpha=0.8, linestyle='-', color='darkgrey')

        if i in first_in_row:
            ax.yaxis.set_visible(True)
            ax.set_ylabel('Scale Factor\n(Multiplier)')
        if i in last_row:
            ax.xaxis.set_visible(True)
            ax.set_xlabel(plt_label)

    legend_kwargs0 = dict(
        bbox_to_anchor=(1.01, 1),
        title=None,
        loc="upper left",
        frameon=True,
        prop=dict(size=10),
    )
    axs[3].legend(**legend_kwargs0)
    # Save and close plot
    plt.subplots_adjust(wspace=0.12, hspace=0.2, top=0.92)
    plt.margins(x=0, y=0)
    plt.suptitle(storm)
    plt.savefig(os.path.join(os.getcwd(), f'scale_factors_by_threshold_{storm}_{var}.png'), bbox_inches='tight',
                dpi=255)
    plt.close()
