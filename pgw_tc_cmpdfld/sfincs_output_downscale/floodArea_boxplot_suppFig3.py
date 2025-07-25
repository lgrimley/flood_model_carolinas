import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
mpl.use('TkAgg')
plt.ion()




os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\03_downscaled\sbgRes200m')
df = pd.read_csv(r'all_runs_stats_sbgRes200m.csv', index_col=0)

# Now back to normal
df = df[['Area_sqkm']]
df['storm'] = [str(x.split('_')[0]) for x in df.index]
df['climate'] = [str(x.split('_')[1]) for x in df.index]
df['group'] = df['storm'] + ' ' + df['climate']
df['attr'] = [str(x.split('_')[-1]) for x in df.index]
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df['attr'] = df['attr'].map(lambda x: mapping.get(x, 'Total'))
df['run'] = df.index.str.replace(r'_attr\d+', '', regex=True)

pivot_df = df.pivot_table(index=['run', 'group','climate'], columns='attr', values='Area_sqkm').reset_index()

# Subset by climate period
pivot_df_pres = pivot_df[pivot_df['climate'] == 'pres']
pivot_df_fut = pivot_df[pivot_df['climate'] == 'fut']

# Plotting info
scenarios = ['Coastal', 'Runoff', 'Compound', 'Total']
nrow = 4
ncol = 1

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

props = dict(boxes="white", whiskers="black", caps="black")
boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)


# PLOTTING Boxplot of flooded area
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, tight_layout=True, figsize=(5, 5))
axes = axes.flatten()
counter = 0
for ax in axes:
    bp = pivot_df_fut.boxplot(ax=ax,
                         by='group',
                         column=scenarios[counter],
                         vert=True,
                         color=props,
                         boxprops=boxprops,
                         flierprops=flierprops,
                         medianprops=medianprops,
                         meanprops=meanpointprops,
                         meanline=False,
                         showmeans=True,
                         patch_artist=True,
                         layout=(3, 1),
                         zorder=1
                         )
    ax.scatter(x=ax.get_xticks(), y=pivot_df_pres[scenarios[counter]].values,
               s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)

    if counter in last_row:
        xtick = ax.get_xticks()
        ax.set_xticklabels(['Flor (n=35)',
                            'Floy (n=35)',
                            'Matt (n=30)'])
    else:
        ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_title(scenarios[counter])
    ax.set_xlabel(None)
    ax.set_ylabel('Flooded Area\n(sq.km.)')
    kwargs = dict(linestyle='--', linewidth=1, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
    ax.set_axisbelow(True)
    counter += 1

plt.suptitle(None)
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig('driver_fldArea_boxplot_ensmean.png', dpi=255)
plt.close()