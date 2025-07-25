import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
mpl.use('TkAgg')
plt.ion()
import pandas as pd
import numpy as np
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\03_downscaled')
df = pd.read_csv(r'downscale_comparsion_depth_fldArea.csv')
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df['attrCode'] = df['attrCode'].map(lambda x: mapping.get(x, 'Total'))

df = df[['storm', 'climate', 'attrCode', 'Area_sqkm', 'gridRes']]
df['group'] = df['storm'].astype(str) + '_' + df['climate'].astype(str)
df.set_index('group', inplace=True, drop=True)


res = 200

dfs = df[df['gridRes'] == res].copy()
dfs.drop(columns=['storm', 'climate', 'gridRes'], inplace=True)
df_reset = dfs.reset_index()  # bring 'group' back as a column to pivot
df_pivoted = df_reset.pivot_table(index='group', columns='attrCode', values='Area_sqkm', aggfunc='first')
df_pivoted.drop(columns=['Total'], inplace=True)
pie_scale = df_pivoted.sum(axis=1) / 8000

# Plotting
runs_to_plot = ['flor_present', 'flor_future', 'floy_present', 'floy_future', 'matt_present', 'matt_future']
storms = ['Florence', 'Floyd', 'Matthew']

# Setup Plot Info
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
colors = ['#4F6272', '#DD7596', '#B7C3F3']
nrow, ncol = 3, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)

fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4, 5))
axs = axs.flatten()
for ax in axs:
    ax.set_aspect('equal')
for i in range(len(runs_to_plot)):
    ax = axs[i]
    d = df_pivoted[df_pivoted.index == runs_to_plot[i]]
    if i in [2,4]:
        wedges, texts, autotexts = ax.pie(d.to_numpy()[0],
               colors=colors,
               radius=pie_scale[pie_scale.index == runs_to_plot[i]][0],
               startangle=90,
               autopct='%1.0f%%',
               pctdistance=1.25,  # Move % labels farther out
               textprops=dict(color="black", fontsize=9)
               )
    else:
        wedges, texts, autotexts = ax.pie(d.to_numpy()[0],
               colors=colors,
               radius=pie_scale[pie_scale.index == runs_to_plot[i]][0],
               startangle=90,
               autopct='%1.0f%%',
               pctdistance=0.65,
               textprops=dict(color="black", fontsize=9)
               )

    # Apply white halo to percentage labels
    for text in autotexts:
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal()
        ])
    if i in [2,3]:
        # Label below each pie
        ax.text(
            0, -1.3,  # x=0 center, y=just below pie
            f"Area: {d.to_numpy().sum():.0f} km²",  # or any custom label
            ha='center',
            va='top',
            transform=ax.transData,
            fontsize=9, #fontweight='bold'
        )
    else:
        # Label below each pie
        ax.text(
            0, -2.1,  # x=0 center, y=just below pie
            f"Area: {d.to_numpy().sum():.0f} km²",  # or any custom label
            ha='center',
            va='top',
            transform=ax.transData,
            fontsize=9, #fontweight='bold'

        )
legend_labels = df_pivoted.columns.tolist()
legend_colors = colors[:len(legend_labels)]  # colors used in pies
handles = [mpatches.Patch(color=c, label=l) for c, l in zip(legend_colors, legend_labels)]
# Add single legend to figure (outside the plot on right)
fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.2, 0.5),
           borderaxespad=0, frameon=True, fontsize=10)
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.3, 0.5, storms[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='horizontal',
                              transform=axs[first_in_row[i]].transAxes,
                              fontsize=10,
                              fontweight='bold'
                              )
c_lab = ['Present', 'Future']
for i in range(len(c_lab)):
    axs[i].text(0.5, 1.5, c_lab[i],
                              horizontalalignment='center',
                              verticalalignment='center',
                              rotation='horizontal',
                              transform=axs[i].transAxes,
                              fontsize=10,
                              fontweight='bold'
                              )
plt.margins(x=0, y=0)
plt.savefig(f'fld_area_pieChart_sbgRes{res}_labels.jpg', dpi=300, bbox_inches='tight')
plt.close()

