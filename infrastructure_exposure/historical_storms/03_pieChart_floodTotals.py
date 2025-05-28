import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True

def adjust_labels(texts, theta, r):
    for i, t in enumerate(texts):
        x = r * np.cos(theta[i])
        y = r * np.sin(theta[i])

        # Adjust horizontal alignment based on which side of the pie the label is on
        ha = 'left' if x >= 0 else 'right'

        # Adjust vertical alignment based on whether the label is above or below the pie
        va = 'bottom' if y >= 0 else 'top'

        t.set_ha(ha)
        t.set_va(va)

        # Move the label slightly away from the pie
        offset = 0.1
        t.set_position((x + np.sign(x) * offset, y + np.sign(y) * offset))

''' 

Get total buildings flooded for historical storms 

'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')

# Load the totals for the buildings
buildings = pd.read_csv('building_counts_across_depthThresh_FlorFloyMatt.csv', index_col=0)
dfb = buildings[buildings['hmin'] == 0.5].copy()
dfb.drop(columns=['hmin', 'Period', 'Coastal-Comp','Runoff-Comp', 'Total'], inplace=True)
dfb['Type'] = 'Flood Extent'

# Load the overland flood extent
fld_extent_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\03_downscaled\downscale_comparsion_depth_fldArea.csv'
fld_extent = pd.read_csv(fld_extent_filepath)
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
fld_extent['attrCode'] = fld_extent['attrCode'].map(lambda x: mapping.get(x, 'Total'))
df = fld_extent[['storm', 'climate', 'attrCode', 'Area_sqkm', 'gridRes']]
df['group'] = df['storm'].astype(str) + '_' + df['climate'].astype(str)
df.set_index('group', inplace=True, drop=True)
dfs = df[df['gridRes'] == 5].copy()
dfs.drop(columns=['storm', 'climate', 'gridRes'], inplace=True)
df_reset = dfs.reset_index()  # bring 'group' back as a column to pivot
df_pivoted = df_reset.pivot_table(index='group', columns='attrCode', values='Area_sqkm', aggfunc='first')
df_pivoted.drop(columns=['Total'], inplace=True)
df_pivoted['Type'] = 'Buildings'

# Combined the building and flood area data before plotting
combined_df = pd.concat(objs=[dfb, df_pivoted],axis=0, ignore_index=False)
combined_df['Storm'] = [x.split('_')[0] for x in combined_df.index]
mapping = {'flor': 'Florence', 'floy': 'Floyd', 'matt': 'Matthew'}
combined_df['Storm'] = combined_df['Storm'].map(lambda x: mapping.get(x))
combined_df['Climate'] = [x.split('_')[-1][0] for x in combined_df.index]
mapping = {'p': 'Present', 'f': 'Future'}
combined_df['Climate'] = combined_df['Climate'].map(lambda x: mapping.get(x))
combined_df = combined_df.reset_index(drop=True)
combined_df.round(0).to_csv('final_totals_for_piechart_FlorFloyMatt.csv')

# Let's try to make this pie plot
plot_storm_pie = True
for storm_name in combined_df['Storm'].unique():
    if plot_storm_pie is True:
        storm_df = combined_df[combined_df['Storm'] == storm_name]
        types = storm_df['Type'].unique()
        climates = ['Present', 'Future']
        pie_labels = ['Coastal', 'Runoff', 'Compound']

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
        # If only one row of type, axes won't be 2D, handle that
        if len(types) == 1:
            axes = axes.reshape(1, -1)
        colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
        for i, t in enumerate(types):
            for j, climate in enumerate(climates):
                ax = axes[i, j]
                # calculate the scale for this pie
                s = storm_df[(storm_df['Type'] == t)]
                scale = np.floor(s[pie_labels].sum(axis=1).mean())

                subset = storm_df[(storm_df['Type'] == t) & (storm_df['Climate'] == climate)]
                ps = subset[pie_labels].sum(axis=1)/scale
                print(ps)
                if not subset.empty:
                    values = subset[pie_labels].values.flatten().astype(np.float32)
                    wedges, texts, autotexts = ax.pie(values,
                           #labels=pie_labels,
                           colors=colors,
                           radius=ps.item(),
                           startangle=90,
                           autopct='%1.0f%%',#pctdistance=0.5
                           )
                    theta = [((w.theta2 + w.theta1) / 2) / 180 * np.pi for w in wedges]
                    if i <2:
                        adjust_labels(texts, theta, 1.3)
                        adjust_labels(autotexts, theta, 0.3)

                else:
                    ax.set_visible(False)

            axes[i,0].text(-0.1, 0.5, types[i],
                           horizontalalignment='right',
                           verticalalignment='center',
                           rotation='vertical',
                           transform=axes[i,0].transAxes)
            axes[0,i].set_title(climates[i], pad=20)

        # Add horizontal legend below all plots
        legend_handles = [Patch(color=colors[i], label=pie_labels[i]) for i in range(3)]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.5, -0.01),  # Adjust vertical space
            frameon=False
        )
        #plt.subplots_adjust(wspace=0.0, hspace=0.0)
        #plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.savefig(f'{storm_name}_pieChart.jpg',dpi=300)






