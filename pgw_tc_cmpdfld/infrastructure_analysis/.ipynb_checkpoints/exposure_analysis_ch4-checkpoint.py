import pandas as pd
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

extent_df = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\process_attribution_mask\stats_for_manuscript.csv', index_col=0)

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')
building_df = pd.read_csv('buildings_tc_exposure_rp_real.csv', index_col=0, low_memory=True)
print(building_df.columns)

depth_threshold = 0.64

counts_list = []
for storm in ['flor','floy','matt']:
    fld_area = extent_df[extent_df.index == f'{storm}_pres']
    fld_build = building_df[building_df[[f'{storm}_compound_hzsmax', f'{storm}_compound_pzsmax']].notna().all(axis=1)]
    #storm_stats = fld_build.describe()

    present_flooded = fld_build[fld_build[f'{storm}_compound_hdepth'] > depth_threshold]
    future_flooded = fld_build[fld_build[f'{storm}_compound_pdepth'] > depth_threshold]

    codes, present_counts = np.unique(present_flooded[f'{storm}_compound_hclass'], return_counts=True)
    codes, future_counts = np.unique(future_flooded[f'{storm}_compound_pclass'], return_counts=True)

    counts_list = counts_list + [present_counts, future_counts]


colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
nrow, ncol = 3, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(6, 6),
                        tight_layout=True,
                        layout='constrained'
                        )
axs = axs.flatten()
for i in range(len(combined.index)):
    ax = axs[i]
    d = combined[combined.index == combined.index[i]]
    wedges, texts, autotexts = ax.pie(d.to_numpy()[0],
           colors=colors,
           radius=pie_scale[pie_scale.index == combined.index[i]][0],
           startangle=90,
           autopct='%1.0f%%',#pctdistance=0.5
           )
    theta = [((w.theta2 + w.theta1) / 2) / 180 * np.pi for w in wedges]
    if i <2:
        adjust_labels(texts, theta, 1.3)
        adjust_labels(autotexts, theta, 0.6)

axs[0].set_title('Present')
axs[1].set_title('Future')
legend_kwargs0 = dict(
    bbox_to_anchor=(0.9, 1.2),
    title=None,
    loc="upper right",
    frameon=True,
    prop=dict(size=10),
)
axs[4].legend(labels=combined.columns, **legend_kwargs0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure\hazard_exposure_Florence_pieChart_legend.jpg', bbox_inches='tight', dpi=300)

