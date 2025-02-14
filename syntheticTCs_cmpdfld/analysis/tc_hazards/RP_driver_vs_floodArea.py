import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True

# Working directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob\basin_data')
data_files = os.listdir()
master_df = pd.DataFrame()
for file in data_files:
    basin = file.split('_')[0]
    period = file.split('_')[-1].split('.')[0]
    data = pd.read_csv(file).astype(float)
    data.rename(columns={'Unnamed: 0': 'tc_id'}, inplace=True)
    data['basin'] = basin
    data['period'] = period
    master_df = pd.concat(objs=[master_df, data], axis=0, ignore_index=True)

print(master_df.columns)
basins = master_df['basin'].unique()
# ['tc_id', 'maxWS', 'meanMaxWS', 'meanWS', 'meanWSthresh',
#        'meanDirection', 'CumPrecipKM3', 'MeanTotPrecipMM', 'MaxTotPrecipMM',
#        'maxRR', 'meanRR', 'meanRRthresh', 'stormtide', 'FldArea', 'vmax',
#        'weight', 'maxWS_rp', 'meanMaxWS_rp', 'meanWS_rp', 'meanWSthresh_rp',
#        'meanDirection_rp', 'CumPrecipKM3_rp', 'MeanTotPrecipMM_rp',
#        'MaxTotPrecipMM_rp', 'maxRR_rp', 'meanRR_rp', 'meanRRthresh_rp',
#        'stormtide_rp', 'FldArea_rp', 'basin', 'period'],

''' PLOTTING Return Period '''
# Plotting prep
bounds = [1, 10, 25, 50, 100, 200, 500]
cmap = plt.cm.Blues
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ckwargs = dict(norm=norm, cmap=cmap)

variable_x = 'MaxTotPrecipMM_rp'
variable_x_label = 'Max TP mm'
variable_y = 'stormtide_rp'
variable_y_label = 'Surge'
variable_z = 'FldArea_rp'

nrow = 5
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4, 7), sharex=True, sharey=True)
for i in range(len(basins)):
    basin = basins[i]
    bd = master_df[master_df['basin'] == basin]
    bdh = bd[bd['period'] == 'historic']
    bdf = bd[bd['period'] == 'future']

    # Historic period
    ax = axs[i][0]
    sc = ax.scatter(bdh[variable_x], bdh[variable_y], c=bdh[variable_z], **ckwargs,
                    s=30, alpha=0.9, edgecolor='grey',linewidths=0.15)
    # Future period
    ax = axs[i][1]
    sc2 = ax.scatter(bdf[variable_x], bdf[variable_y], c=bdf[variable_z], **ckwargs,
                    s=30, alpha=0.9, edgecolor='grey',linewidths=0.15)

    axs[i][0].text(-0.6, 0.5, basin, horizontalalignment='right', verticalalignment='center',
                   rotation='vertical', transform=axs[i][0].transAxes)
axs = axs.flatten()
for i in range(len(axs)):
    ax = axs[i]
    ax.set_yscale('log')
    ax.set_ylim(1, 2000)
    ax.set_xscale('log')
    ax.set_xlim(1, 2000)
    ax.grid('both')
    ax.set_aspect('equal', adjustable='box')
    if i in last_row:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(variable_x_label)
    if i in first_in_row:
        ax.set_ylabel(variable_y_label)
    # Define the desired tick positions and labels
    ticks = [1, 10, 100, 500]
    labels = ['1', '10', '100', '500']
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

axs[0].set_title('Historic (1980-2005)')
axs[1].set_title('Projected (2070-2100)')
pos0 = axs[5].get_position()
cax1 = fig.add_axes([pos0.x1 + 0.15, pos0.y0, 0.03, pos0.height * 1.5])
cbar1 = fig.colorbar(sc, cax=cax1, orientation='vertical',
                     label='Compound Flood Extent\nReturn Period',
                     extend='max')
plt.margins(x=0, y=0)
plt.savefig(fr'..\{variable_y}_vs_{variable_x}_{variable_z}.png',
            dpi=300,
            bbox_inches="tight"
            )
plt.close()
