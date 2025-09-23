import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

mpl.use('TkAgg')
plt.ion()

###################################################################################################################
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\01_return_period_tables')
outdir = r'..\06_copula\correlation_matrices'

# sel_cols = ['maxWS', 'meanMaxWS', 'meanWSthresh', 'CumPrecipKM3', 'MeanTotPrecipMM', 'MaxTotPrecipMM', 'maxRR',
#        'AvgmaxRR', 'meanRRthresh', 'stormtide', 'Coastal_Area_sqkm', 'Compound_Area_sqkm',
#        'Runoff_Area_sqkm', 'Total_Area_sqkm', 'basin']

sel_cols = ['maxWS', 'meanMaxWS',
            'MeanTotPrecipMM', 'MaxTotPrecipMM', 'maxRR', 'AvgmaxRR',
            'stormtide', 'Coastal_Area_sqkm', 'Compound_Area_sqkm', 'Runoff_Area_sqkm', 'Total_Area_sqkm', 'basin']
###################################################################################################################
ncep_csvfiles = [f for f in os.listdir() if f.endswith('ncep.csv')]
histdf = pd.concat((pd.read_csv(file, index_col=0) for file in ncep_csvfiles), ignore_index=False)
cols_to_check = ['Coastal_Area_sqkm', 'Compound_Area_sqkm', 'Runoff_Area_sqkm', 'Total_Area_sqkm']
percentiles = histdf[cols_to_check].quantile(0.9)
condition = histdf[cols_to_check].gt(percentiles)
subset_df = histdf[condition.any(axis=1)]
histdf = histdf[sel_cols]

# Track info at landfall
track_table_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\tracks\ncep_landfall_track_info.csv'
track_df = pd.read_csv(track_table_filepath, index_col=0)
track_df.set_index('tc_id', inplace=True, drop=True)
track_df_hist = track_df[['rmw100','pstore100','speed100','vstore100']]
#histdf = histdf.merge(track_df_hist, left_index=True, right_index=True, how='left')

###################################################################################################################
# Load the projected/future TC data
canesm_csvfiles = [f for f in os.listdir() if f.endswith('canesm.csv')]
futdf = pd.concat((pd.read_csv(file, index_col=0) for file in canesm_csvfiles), ignore_index=False)
futdf = futdf[sel_cols]

# Track info at landfall
track_table_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks\canesm_landfall_track_info.csv'
track_df = pd.read_csv(track_table_filepath, index_col=0)
track_df.set_index('tc_id', inplace=True, drop=True)
track_df_fut = track_df[['rmw100','pstore100','speed100','vstore100']]

###################################################################################################################

##### CHANGE ME HERE ##########
df = histdf
clim = 'NCEP'
track_df = track_df_hist
#############################

sections = {
    'Track': ['rmw100','pstore100','speed100','vstore100'],
    'Wind': ['maxWS', 'meanMaxWS'],
    'Rainfall': ['MeanTotPrecipMM', 'MaxTotPrecipMM', 'maxRR', 'AvgmaxRR'],
    'Hazard': ['stormtide', 'Coastal_Area_sqkm', 'Compound_Area_sqkm', 'Runoff_Area_sqkm','Total_Area_sqkm'],
}
display_names = {
    'rmw100': 'RMW',
    'pstore100': 'MCP',
    'speed100': 'TS',
    'vstore100': 'Max SWS',
    'maxWS': 'Max WS',
    'meanMaxWS': 'Avg Max WS',
    'MeanTotPrecipMM': 'Avg TP',
    'MaxTotPrecipMM': 'Max TP',
    'maxRR': 'Max RR',
    'AvgmaxRR': 'Avg Max RR',
    'stormtide': 'Storm Tide',
    'Coastal_Area_sqkm': 'Coastal',
    'Runoff_Area_sqkm': 'Runoff',
    'Compound_Area_sqkm': 'Compound',
    'Total_Area_sqkm': 'Total'
}
group_sizes = [len(v) for v in sections.values()]
ordered_cols = [col for group in sections.values() for col in group]
pval_lim = 0.05

rho_matrix_list = []
pval_matrix_list = []
# --- Main loop ---
for basin_name, group in df.groupby('basin'):
    print(f"\n🔹 Basin: {basin_name}")
    group = group.drop(columns='basin', errors='ignore')
    group = pd.concat([group, track_df], axis=1)


    group = group.select_dtypes(include='number').dropna()
    cols = group.columns

    # Compute rho and p-value
    rho_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            rho, pval = scipy.stats.spearmanr(group[col1], group[col2], nan_policy='omit')
            rho_matrix.loc[col1, col2] = rho
            pval_matrix.loc[col1, col2] = pval

    # Reorder and filter for significance
    rho_matrix = rho_matrix.loc[ordered_cols, ordered_cols]
    pval_matrix = pval_matrix.loc[ordered_cols, ordered_cols]

    rho_matrix.to_csv(os.path.join(outdir, f'spearman_correlation_{basin_name}_{clim}.csv'))
    pval_matrix.to_csv(os.path.join(outdir, f'spearman_correlation_{basin_name}_{clim}_pvalue.csv'))

    rho_matrix_list.append(rho_matrix)
    pval_matrix_list.append(pval_matrix)

    # Plotting below
    sig_rho = rho_matrix.mask(pval_matrix >= pval_lim)
    mask = np.triu(np.ones_like(sig_rho, dtype=bool), k=1)
    xticklabels = [display_names.get(col, col) for col in ordered_cols]
    yticklabels = [display_names.get(col, col) for col in ordered_cols]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(sig_rho, mask=mask,cmap='coolwarm', annot=True, fmt=".2f", center=0, ax=ax,
                     annot_kws={'size': 7},
                     linewidths=1, linecolor='white', cbar=False, square=False,
                     vmin=-1, vmax=1,
                     xticklabels=xticklabels, yticklabels=yticklabels,
                     #cbar_kws={'shrink': 0.8, 'location':'bottom'}
                     )
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    # Adjust size and position (left, bottom, width, height)
    cax = divider.append_axes("top", size="4%", pad=0.5)  # 5% height, 0.7 pad above heatmap

    # Create colorbar
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])  # needed for ScalarMappable

    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(f'Correlation Coefficient (p < {pval_lim})')

    # Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=8)

    # Add section group labels (without spacing)
    section_positions = np.cumsum([0] + group_sizes[:-1])
    for i, (label, start) in enumerate(zip(sections.keys(), section_positions)):
        group_len = group_sizes[i]
        mid = start + group_len / 2 - 0.5
        n = len(ordered_cols)
        # Add labels outside the heatmap
        ax.text(-3, mid, label, va='center', ha='center', fontsize=8, fontweight='bold', rotation=90)
        ax.text(mid, n + 5, label, va='bottom', ha='center', fontsize=8, fontweight='bold', rotation=0)

    # # Section start positions
    # section_starts = np.cumsum([0] + group_sizes[:-1])
    #
    # # Draw box around each section block (diagonal squares)
    # for start, size in zip(section_positions, group_sizes):
    #     rect = patches.Rectangle(
    #         (start, start),
    #         width=size,
    #         height=size,
    #         fill=False,
    #         edgecolor='black',
    #         linewidth=2
    #     )
    #     ax.add_patch(rect)

    # Titles and output
    #title = f"{basin_name} ({clim}): Spearman Correlation (p < {pval_lim})"
    #plt.title(title, fontsize=10)
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for title
    plt.savefig(os.path.join(outdir, f'spearman_correlation_{basin_name}_{clim}_pvalue{pval_lim}_mask.png'), dpi=300, bbox_inches='tight')
    plt.close()


vars = ['Compound_Area_sqkm', 'Runoff_Area_sqkm','Coastal_Area_sqkm','Total_Area_sqkm']
for v in vars:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes = axes.flatten()
    basins = df['basin'].unique()
    for i in range(len(basins)):
        basin_name = basins[i]
        data = df[df['basin'] == basin_name]
        data = pd.concat([data, track_df], axis=1)

        data = data.select_dtypes(include='number').dropna()
        data_sorted = data.sort_values(by=v)
        log_colors = np.log10(data_sorted[v])

        ax = axes[i]
        sc = ax.scatter(x=data_sorted['speed100'],
                         #y=data_sorted['rmw100'],
                        y=data_sorted['vstore100'],
                         c=log_colors,
                         cmap='Reds'
                         )

        ax.set_title(basin_name)
        # Set x/y labels only on bottom row and left column
        if i >= len(axes) - 2:  # bottom row
            ax.set_xlabel('Translation Speed (m/s)')
        if i % 2 == 0:  # left column
            #ax.set_ylabel('Radius of Max Winds (km)')
            ax.set_ylabel('Max Sustained\nWind Speed (knots)')

    # Add a single colorbar for all subplots
    fig.colorbar(sc, ax=axes, location='right', label=f'log10({v})')
    plt.show()
    plt.savefig(f'{v}.png', dpi=300)
    plt.close()


