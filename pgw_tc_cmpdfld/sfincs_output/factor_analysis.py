import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

area = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis\driver_analysis'
                   r'\pgw_drivers_classified_all_area.csv', index_col=0)
area.drop(columns=['no_flood'])
area['storm'] = [i.split("_")[0] for i in area.index]

present_ensmean = area[area.index.isin(['flor_pres', 'floy_pres', 'matt_pres'])]

area = area[~area.index.isin(['flor_pres', 'floy_pres', 'matt_pres'])]
area['sf'] = [i.split("_")[-2] for i in area.index]
area['slr'] = [i.split("_")[-1] for i in area.index]

rel_change_factor = pd.DataFrame()
for storm in area['storm'].unique():
    rcf_storm = pd.DataFrame()
    pres_area = present_ensmean[present_ensmean['storm'] == storm]
    for index, row in area[area['storm'] == storm].T.iterrows():
        if index in ['coastal', 'compound', 'runoff']:
            p = pres_area[index].values.item()  # present climate fld area
            rcf = ((row - p) / p) * 100  # percent increase in fld area in future
            rcf = pd.DataFrame(rcf)
            rcf.columns = [f'{index}_pct']
            rcf_storm = pd.concat([rcf_storm, rcf.astype(float)], axis=1)
    rel_change_factor = pd.concat([rel_change_factor, rcf_storm], axis=0)
master_df = pd.concat([area, rel_change_factor], axis=1)

slr = pd.read_csv(
    r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\hindcast_slr_event_ids.csv',
    index_col=0, header=None)
slr.columns = ['slr_m']
slr['slr_rcf'] = 1 + ((slr['slr_m'] - 0.13) / 0.13)
master_df = pd.concat([master_df, slr.astype(float)], axis=1)
master_df.dropna(how='any', axis=0, inplace=True)

rain_sf = pd.read_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors'
                      r'\presScaled_precip_multiplier.csv', index_col=0, header=None)
rain_sf.columns = ['rain_sf']
rain_sf = rain_sf[~rain_sf.index.isin(['flor_ensmean', 'floy_ensmean', 'matt_ensmean'])]
rain_sf['storm'] = [i[0] for i in rain_sf.index.str.split('_')]
rain_sf['sf_id'] = [f'SF{i[-1][-1][-1]}' for i in rain_sf.index.str.split('_')]

precip_ss = []
for index in master_df.index:
    d = master_df.loc[index]
    sf = d['sf']
    storm = d['storm']
    ss = rain_sf[(rain_sf['storm'] == storm)]
    ss = ss[ss['sf_id'] == sf]
    ss = (ss['rain_sf'].values.item() - 1) * 100
    precip_ss.append(float(ss))
master_df['precip_rcf'] = precip_ss
master_df = master_df.round(3)

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
marker = ['o', 's', 'd']
storms = ['flor', 'floy', 'matt']

# Coastal flood area vs. sea level rise plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
for i in range(len(storms)):
    da_plot = master_df[master_df['storm'] == storms[i]]
    x = da_plot['slr_m']
    y = da_plot['coastal'] / 1000
    coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x, poly1d_fn(x),
            color='black',
            linestyle='-',
            linewidth=0.75,
            zorder=0)
    sp = ax.scatter(x=x,
                    y=y,
                    marker=marker[i],
                    edgecolors='black',
                    label=storms[i],
                    zorder=1
                    )
ax.legend(loc='best')
ax.set_xlabel('Sea Level Rise (m)')
ax.set_ylabel('Coastal Flood Area\n(thousand sq.km.)')
ax.set_xlim(0.65, 1.6)
ax.set_ylim(6.5, 11)
plt.subplots_adjust(wspace=0, hspace=0.97)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\coastal_fldArea_vs_SLR.png',
            dpi=255,
            bbox_inches="tight")
plt.close()

# Runoff flood area vs. precip scale plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
for i in range(len(storms)):
    da_plot = master_df[master_df['storm'] == storms[i]]
    x = da_plot['precip_rcf']
    y = da_plot['runoff'] / 1000
    coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x, poly1d_fn(x),
            color='black',
            linestyle='-',
            linewidth=0.75,
            zorder=0)
    sp = ax.scatter(x=x,
                    y=y,
                    marker=marker[i],
                    edgecolors='black',
                    label=storms[i],
                    zorder=1
                    )
ax.legend(loc='best')
ax.set_xlabel('Increase in Mean Rain Rate (%)')
ax.set_ylabel('Runoff Flood Area\n(thousand sq.km.)')
ax.set_xlim(14, 55)
ax.set_ylim(60, 64)
plt.subplots_adjust(wspace=0, hspace=0.97)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\runoff_fldArea_vs_precipScale.png',
            dpi=255,
            bbox_inches="tight")
plt.close()

# compound vs. SLR vs. rain
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 6), sharex=True, sharey=False)
axs = axs.flatten()
colors = ['black', 'black', 'black']
for i in range(len(storms)):
    ax = axs[i]
    da_plot = master_df[master_df['storm'] == storms[i]]
    y = da_plot['precip_rcf']
    x = da_plot['slr_m']
    coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x, poly1d_fn(x),
            color=colors[i],
            linestyle='-',
            linewidth=1,
            zorder=0)
    sp = ax.scatter(x=da_plot['slr_m'],
                    y=da_plot['precip_rcf'] - 1,
                    c=da_plot['compound'] / 1000,
                    cmap='Reds',
                    marker=marker[i],
                    edgecolors=colors[i],
                    zorder=1
                    )
    ax.set_ylabel('% Increase in\nMean Rain Rate')
    ax.set_title('')
    ax.set_title(f'{storms[i]}')
ax.set_xlabel('Sea Level Rise (m)')
pos0 = axs[1].get_position()  # get the original position
cax1 = fig.add_axes([pos0.x1 + 0.1, pos0.y0, 0.05, pos0.height * 1.2])
cbar1 = fig.colorbar(sp,
                     cax=cax1,
                     orientation='vertical',
                     #ticks=[2, 4, 6],
                     extend='both',
                     label='Compound Flood Area\n(thousand sk.km.))'
                     )
plt.subplots_adjust(wspace=0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\compound_vs_CC.png',
            dpi=255,
            bbox_inches="tight")
plt.close()
