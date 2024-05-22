import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

slr = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\slr_event_ids_DO_NOT_DELETE.csv',
                  header=None)
slr.columns = ['event_id', 'slr_m']
slr['storm'] = [name.split('_')[0] for name in slr['event_id']]
slr['run'] = [name.split('_')[-2] for name in slr['event_id']]

# PLOTTING Boxplot of flooded area
# This script reads in the sea level rise and rain rate data used in the future scenarios

storms = ['flor', 'floy', 'matt']
fig, axes = plt.subplots(nrows=3, ncols=1, tight_layout=True, figsize=(4, 5), sharey=True, sharex=False)
counter = 0
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlabel(None)
    ax.set_title(f'{storms[counter]}')
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('SLR (m)')
    ax.set_ylim((0.5, 1.8))
    kwargs = dict(linestyle='-', linewidth=0.75, color='grey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
    ax.set_axisbelow(True)

    counter += 1

plt.suptitle(None)
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
# plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\slr_boxplot_by_storm_by_run.png',
#             tight_layout=True,
#             bbox_inches='tight', dpi=255)
plt.close()

# RAINFALL
# rr = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors\scale_factors_carolinas'
#                  r'\precip_thresh_5_to_100\precip_thresh_5_to_100_data.csv', header=0)
# climate = 'future'
# rr = rr[rr['climate'] == climate]
# rr_ensmean = rr[rr['run'] == 'ensmean']
# rr_members = rr[rr['run'] != 'ensmean']
# PLOTTING rain rates
# fig, axes = plt.subplots(nrows=2, ncols=2, tight_layout=True, figsize=(5, 4))
# axes = axes.flatten()
# ds = rr_members
# c = ['mean', '50%', '75%', '99%']
# for counter in range(len(c)):
#     ax = axes[counter]
#     bp = ds.boxplot(ax=ax,
#                     by='storm',
#                     column=c[counter],
#                     vert=True,
#                     color=props,
#                     boxprops=boxprops,
#                     flierprops=flierprops,
#                     medianprops=medianprops,
#                     meanprops=meanpointprops,
#                     meanline=False,
#                     showmeans=True,
#                     patch_artist=True,
#                     layout=(1, 1),
#                     zorder=1
#                     )
#     ensmean = rr_ensmean[c[counter]].values
#     ax.scatter(x=ax.get_xticks(), y=ensmean,
#                s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
#     ax.set_ylim(min(ensmean) * 0.95, max(ds[c[counter]]) * 1.03)
#     # ax.set_aspect('equal')
#     ax.set_xlabel(None)
#     # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
#     ax.set_ylabel('Rain Rate (mm/hr)')
#     # ax.set_ylim((11, 17))
#     kwargs = dict(linestyle='-', linewidth=0.75, color='grey', alpha=0.8)
#     ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
#     kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
#     ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
#     ax.set_axisbelow(True)
# 
# plt.suptitle(f'{climate}: Carolinas Bounding Area')
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.margins(x=0, y=0)
# plt.savefig(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\rr_boxplot_by_storm_{climate}.png',
#             tight_layout=True,
#             bbox_inches='tight', dpi=255)
# plt.close()

# Scale factors
rr = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors\scale_factors_carolinas'
                 r'\precip_thresh_5_to_100\precip_thresh_5_to_100_data.csv', header=0)
rr_fut = rr[rr['climate'] == 'future']
rr_fut_ensmean = rr_fut[rr_fut['run'] == 'ensmean']
rr_fut_members = rr_fut[rr_fut['run'] != 'ensmean']
rr_pres = rr[rr['climate'] == 'present']
# rr_pres['fut_minus_pres'] = ((rr_fut['mean'].values - rr_pres['mean'].values) / rr_pres['mean'].values) * 100
rr_pres_ensmean = rr_pres[rr_pres['run'] == 'ensmean']
rr_pres_members = rr_pres[rr_pres['run'] != 'ensmean']

ws = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors'
                 r'\scale_factors_carolinas_landMask\wind_spd_thresh_10_to_100\wind_spd_thresh_10_to_100_data.csv',
                 header=0)
ws_fut = ws[ws['climate'] == 'future']
ws_fut_ensmean = ws_fut[ws_fut['run'] == 'ensmean']
ws_fut_members = ws_fut[ws_fut['run'] != 'ensmean']
ws_pres = ws[ws['climate'] == 'present']
# rr_pres['fut_minus_pres'] = ((rr_fut['mean'].values - rr_pres['mean'].values) / rr_pres['mean'].values) * 100
ws_pres_ensmean = ws_pres[ws_pres['run'] == 'ensmean']
ws_pres_members = ws_pres[ws_pres['run'] != 'ensmean']
rr_pres_ensmean.sort_values(by='storm', ascending=True, inplace=True)
ws_pres_ensmean.sort_values(by='storm', ascending=True, inplace=True)

plot_boxplot = True
if plot_boxplot is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

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
    plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\future_sf_boxplot_noWRFensmean.png',
                bbox_inches='tight',
                tight_layout=True,
                dpi=255)
    plt.close()
