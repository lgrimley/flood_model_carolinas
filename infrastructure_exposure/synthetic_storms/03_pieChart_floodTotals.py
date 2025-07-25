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
buildings = pd.read_csv('building_counts_across_depthThresh_100yr.csv', index_col=0)
dfb = buildings[buildings['hmin'] == 0.5].copy()
dfb.drop(columns=['hmin', 'Period', 'Coastal-Comp','Runoff-Comp', 'Total'], inplace=True)
dfb['Type'] = 'Buildings'
dfb.index = ['RP100_Historic','RP100_Future']


# Load the overland flood extent
wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound'
csvfiles = [f for f in os.listdir(wdir) if f.endswith('.csv')]

# Present AEP
df1 = pd.read_csv(os.path.join(wdir, csvfiles[1]), index_col=0)
df1.index = df1.index.str.replace(' ', '', regex=False)
df1 = df1[~df1.index.duplicated(keep='last')]
df1.index = df1.index.str.split('_', expand=True)
df1.index.names = ['rp','basin','attr']
df1.reset_index(inplace=True)
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df1['attr'] = df1['attr'].map(lambda x: mapping.get(x, 'Total'))
df1['Climate'] = 'Historic'
df_hist = df1

# Future AEP
df1 = pd.read_csv(os.path.join(wdir,csvfiles[0]), index_col=0)
df1.index = df1.index.str.replace(' ', '', regex=False)
df1 = df1[~df1.index.duplicated(keep='last')]
df1.index = df1.index.str.split('_', expand=True)
df1.index.names = ['rp','basin','attr']
df1.reset_index(inplace=True)
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df1['attr'] = df1['attr'].map(lambda x: mapping.get(x, 'Total'))
df1['Climate'] = 'Future'
df_fut = df1

# organize flood extent data
fld_extent_df = pd.concat(objs=[df_hist,df_fut], axis=0, ignore_index=True)
fld_extent = fld_extent_df[(fld_extent_df['rp'] == 'RP100') & (fld_extent_df['basin'] == 'Domain')]
df = fld_extent[['rp', 'Climate', 'attr', 'Area_sqkm']].copy()
df['group'] = df['rp'].astype(str) + '_' + df['Climate'].astype(str)
df.set_index('group', inplace=True, drop=True)
#df.drop(columns=['rp, Climate'], inplace=True)
df_reset = df.reset_index()  # bring 'group' back as a column to pivot
df_pivoted = df_reset.pivot_table(index='group', columns='attr', values='Area_sqkm', aggfunc='first')
df_pivoted.drop(columns=['Total'], inplace=True)
df_pivoted['Type'] = 'Flood Extent'

# Combined the building and flood area data before plotting
combined_df = pd.concat(objs=[dfb, df_pivoted],axis=0, ignore_index=False)
combined_df.round(0).to_csv('final_totals_for_piechart_100yr.csv')

# Let's try to make this pie plot
storm_df = combined_df
storm_df['Climate'] = [x.split('_')[-1] for x in storm_df.index]
plot_storm_pie = True
if plot_storm_pie is True:
    types = storm_df['Type'].unique()
    climates = ['Historic', 'Future']
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
    plt.savefig(f'100yr_pieChart.jpg',dpi=300)






