import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis')
# slr = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\slr_event_ids_DO_NOT_DELETE.csv',
#                   header=None)
# slr.columns = ['event_id', 'slr_m']
# slr['storm'] = [name.split('_')[0] for name in slr['event_id']]
# slr['run'] = [name.split('_')[-2] for name in slr['event_id']]


storms = ['flor', 'floy', 'matt']

''' RAIN RATE'''
rr = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors\scale_factors_carolinas'
                 r'\precip_thresh_5_to_100\precip_thresh_5_to_100_data.csv', header=0)
rr_fut = rr[rr['climate'] == 'future']
rr_fut_ensmean = rr_fut[rr_fut['run'] == 'ensmean']
rr_fut_members = rr_fut[rr_fut['run'] != 'ensmean']
rr_pres = rr[rr['climate'] == 'present']
rr_pres_ensmean = rr_pres[rr_pres['run'] == 'ensmean']
rr_pres_members = rr_pres[rr_pres['run'] != 'ensmean']

combined = pd.concat([rr_pres_members, rr_fut_members], axis=0, ignore_index=True)
combined.set_index(['run_id'], inplace=True)

# Calculate std of the mean
stats = pd.DataFrame()
for storm in ['florence', 'floyd', 'matthew']:
    stats_list = [f'{storm}']

    runs1 = [x for x in combined.index if storm in x and 'present' in x]
    d1 = combined[combined.index.isin(runs1)]

    runs2 = [x for x in combined.index if storm in x and 'future' in x]
    d2 = combined[combined.index.isin(runs2)]

    for s in ['mean', '50%', '90%']:
        diff = d2[stat].values - d1[stat].values
        std = np.std(diff)
        stats_list.append(diff.mean())
        stats_list.append(std)
    stats_df = pd.DataFrame(stats_list).T
    stats_df.columns = ['event', 'mean', 'mean_std', '50%', '50%_std', '90%', '90%_std']
    stats = pd.concat([stats, stats_df], axis=0)
stats.set_index('event', inplace=True, drop=True)
stats = stats.astype(float).round(2)

''' WIND SPEED '''
ws = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors'
                 r'\scale_factors_carolinas_landMask\wind_spd_thresh_10_to_100\wind_spd_thresh_10_to_100_data.csv',
                 header=0)
ws_fut = ws[ws['climate'] == 'future']
ws_fut_ensmean = ws_fut[ws_fut['run'] == 'ensmean']
ws_fut_members = ws_fut[ws_fut['run'] != 'ensmean']
ws_pres = ws[ws['climate'] == 'present']
ws_pres_ensmean = ws_pres[ws_pres['run'] == 'ensmean']
ws_pres_members = ws_pres[ws_pres['run'] != 'ensmean']

combined = pd.concat([ws_pres_members, ws_fut_members], axis=0, ignore_index=True)
combined.set_index(['run_id'], inplace=True)
# Calculate std of the mean
stats = pd.DataFrame()
for storm in ['florence', 'floyd', 'matthew']:
    stats_list = [f'{storm}']

    runs1 = [x for x in combined.index if storm in x and 'present' in x]
    d1 = combined[combined.index.isin(runs1)]

    runs2 = [x for x in combined.index if storm in x and 'future' in x]
    d2 = combined[combined.index.isin(runs2)]

    for s in ['mean', '50%', '90%']:
        diff = d2[stat].values - d1[stat].values
        std = np.std(diff)
        stats_list.append(diff.mean())
        stats_list.append(std)
    stats_df = pd.DataFrame(stats_list).T
    stats_df.columns = ['event', 'mean', 'mean_std', '50%', '50%_std', '90%', '90%_std']
    stats = pd.concat([stats, stats_df], axis=0)
stats.set_index('event', inplace=True, drop=True)
stats = stats.astype(float).round(2)

''' PLOT Details '''
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
props = dict(boxes="white", whiskers="black", caps="black")
boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

plot_boxplot = True
if plot_boxplot is True:
    ds_plot = [rr_pres_members[['mean', 'storm']],
               rr_fut_members[['mean', 'storm']],
               ws_pres_members[['mean', 'storm']],
               ws_fut_members[['mean', 'storm']]]

    nrow = 2
    ncol = 2
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    # PLOTTING Boxplot of flooded area
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, tight_layout=True, figsize=(5, 4))
    axes = axes.flatten()
    for ii in range(len(ds_plot)):
        ax = axes[ii]
        bp = ds_plot[ii].boxplot(ax=ax,
                                 by='storm',
                                 vert=True,
                                 color=props,
                                 boxprops=boxprops,
                                 flierprops=flierprops,
                                 medianprops=medianprops,
                                 meanprops=meanpointprops,
                                 meanline=False,
                                 showmeans=True,
                                 patch_artist=True,
                                 zorder=1,
                                 )
        ax.set_xlabel(None)
        ax.set_title('')
        if ii in first_row:
            ax.set_ylabel('Mean Rain Rate\n(mm/hr)')
            ax.set_ylim(12, 20)
            ax.xaxis.set_tick_params(labelbottom=False)
        if ii in last_row:
            ax.set_ylabel('Mean Wind Speed\n(m/s)')
            xtick = ax.get_xticks()
            ax.set_xticklabels(['Flor\n(n=7)', 'Floy\n(n=7)', 'Matt\n(n=6)'])
            ax.set_ylim(15.5, 22.5)
        if ii in last_in_row:
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(labelbottom=False, labelleft=False)

    axes[0].set_title('Present')
    axes[1].set_title('Future')

    for ax in axes:
        ax.set_xlabel('')
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)

    plt.suptitle(None)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(r'future_TC_vars_boxplot.png', bbox_inches='tight', dpi=255)
    plt.close()

var_stats = pd.DataFrame()
tracker = []
event_id = ['pres_rr', 'fut_rr', 'pres_ws', 'fut_ws']
counter = 0
for ds in ds_plot:
    tracker.append(event_id[counter])
    df = ds.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
    var_stats = pd.concat([var_stats, df], ignore_index=False)
    for storm in ds.storm.unique():
        ds2 = ds[ds['storm'] == storm]
        df = ds2.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
        var_stats = pd.concat([var_stats, df], ignore_index=False)
        tracker.append(f'{event_id[counter]}_{storm}')
    counter += 1
var_stats.index = tracker
var_stats.to_csv('wrf_var_stats.csv')

plot_boxplot_withEnsmean = False
if plot_boxplot_withEnsmean is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

    # WRF Ensemble Mean
    rr_pres_ensmean.sort_values(by='storm', ascending=True, inplace=True)
    ws_pres_ensmean.sort_values(by='storm', ascending=True, inplace=True)

    # Drop ensemble means from dataframe
    # da_slr_plot = slr
    # mask = da_slr_plot['run'] != 'ensmean'
    # da_slr_plot['run'][mask] = 'ensemble'
    # da_slr_plot['group'] = da_slr_plot['storm'] + ' ' + da_slr_plot['run']
    # Organize dataframe of ensemble means

    # PLOTTING Boxplot of flooded area
    fig, axes = plt.subplots(nrows=2, ncols=2, tight_layout=True, figsize=(5, 4))
    axes = axes.flatten()
    ax = axes[0]
    bp = rr_pres_members[['mean', 'storm']].boxplot(ax=ax,
                                                    by='storm',
                                                    vert=True,
                                                    color=props,
                                                    boxprops=boxprops,
                                                    flierprops=flierprops,
                                                    medianprops=medianprops,
                                                    meanprops=meanpointprops,
                                                    meanline=False,
                                                    showmeans=True,
                                                    patch_artist=True,
                                                    zorder=1,
                                                    # layout=(1,2)
                                                    )
    xtick = ax.get_xticks()
    # ax.scatter(x=xtick, y=rr_pres_ensmean['mean'].values,
    #            s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
    ax.set_xticklabels(['Flor\n(n=7)', 'Floy\n(n=7)', 'Matt\n(n=6)'])
    ax.set_xlabel(None)
    ax.set_title('Present Mean Rain Rate')
    ax.set_ylabel('mm/hr')
    ax.set_ylim(9, 20)

    ax = axes[1]
    bp1 = rr_fut_members[['mean', 'storm']].boxplot(ax=ax,
                                                    by='storm',
                                                    vert=True,
                                                    color=props,
                                                    boxprops=boxprops,
                                                    flierprops=flierprops,
                                                    medianprops=medianprops,
                                                    meanprops=meanpointprops,
                                                    meanline=False,
                                                    showmeans=True,
                                                    patch_artist=True,
                                                    zorder=1
                                                    )
    # ax.scatter(x=xtick, y=rr_fut_ensmean['mean'].values,
    #            s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
    ax.set_xticklabels(['Flor\n(n=7)', 'Floy\n(n=7)', 'Matt\n(n=6)'])
    ax.set_xlabel(None)
    ax.set_title('Future Mean Rain Rate')
    ax.set_ylabel('mm/hr')
    ax.set_ylim(9, 20)

    ax = axes[2]
    bp2 = ws_pres_members[['mean', 'storm']].boxplot(ax=ax,
                                                     by='storm',
                                                     vert=True,
                                                     color=props,
                                                     boxprops=boxprops,
                                                     flierprops=flierprops,
                                                     medianprops=medianprops,
                                                     meanprops=meanpointprops,
                                                     meanline=False,
                                                     showmeans=True,
                                                     patch_artist=True,
                                                     zorder=1
                                                     )
    # xtick = ax.get_xticks()
    # ax.scatter(x=xtick, y=ws_pres_ensmean['mean'].values,
    #            s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
    ax.set_xticklabels(['Flor\n(n=7)', 'Floy\n(n=7)', 'Matt\n(n=6)'])
    ax.set_xlabel(None)
    ax.set_title('Present Mean Wind Speed')
    ax.set_ylabel('m/s')
    ax.set_ylim(15, 23)

    ax = axes[3]
    bp3 = ws_fut_members[['mean', 'storm']].boxplot(ax=ax,
                                                    by='storm',
                                                    vert=True,
                                                    color=props,
                                                    boxprops=boxprops,
                                                    flierprops=flierprops,
                                                    medianprops=medianprops,
                                                    meanprops=meanpointprops,
                                                    meanline=False,
                                                    showmeans=True,
                                                    patch_artist=True,
                                                    zorder=1
                                                    )
    # ax.scatter(x=xtick, y=ws_fut_ensmean['mean'].values,
    #            s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
    ax.set_xticklabels(['Flor\n(n=7)', 'Floy\n(n=7)', 'Matt\n(n=6)'])
    ax.set_xlabel(None)
    ax.set_title('Future Mean Wind Speed')
    ax.set_ylabel('m/s')
    ax.set_ylim(15, 23)
    # bp2 = slr.boxplot(ax=ax,
    #                   by='storm',
    #                   column='slr_m',
    #                   vert=True,
    #                   color=props,
    #                   boxprops=boxprops,
    #                   flierprops=flierprops,
    #                   medianprops=medianprops,
    #                   meanprops=meanpointprops,
    #                   meanline=False,
    #                   showmeans=True,
    #                   patch_artist=True,
    #                   zorder=1
    #                   )
    # ax.set_ylabel('m')
    # ax.set_xticks(ax.get_xticks(),
    #               labels=['Flor\n(n=40)',
    #                       'Floy\n(n=40)',
    #                       'Matt\n(n=35)'],
    #               #rotation=0, ha='right'
    #               )
    # ax.set_title('Sea Level Rise')
    # ax.set_ylim(0.5, 1.75)
    # props2 = dict(edgecolor='r')
    # axes[2].findobj(mpl.patches.Patch)[0].update(props2)
    # axes[2].findobj(mpl.patches.Patch)[2].update(props2)
    # axes[2].findobj(mpl.patches.Patch)[4].update(props2)

    for ax in axes:
        ax.set_xlabel('')
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)
    plt.suptitle(None)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(
        r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\future_sf_boxplot_noWRFensmean.png',
        bbox_inches='tight',
        tight_layout=True,
        dpi=255)
    plt.close()
