import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True


wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound\tables'
os.chdir(wdir)
fut_df = pd.read_csv(r'canesm_ssp585_AEP_floodStats_compound_sbgRes5m_hmin0.05m.csv', index_col=0)
past_df = pd.read_csv(r'ncep_AEP_floodStats_compound_sbgRes5m_hmin0.05m.csv', index_col=0)

# Common function to preprocess
def preprocess_df(df):
    df = df.reset_index().rename(columns={'index': 'Label'})
    df[['RP', 'Area', 'Attr']] = df['Label'].str.extract(r'RP(\d+)_([A-Za-z]+)(?:_attr(\d))?')
    df['Flood_Extent'] = df['Attr'].map({'1': 'Coastal', '2': 'Compound', '3': 'Runoff'})
    df['RP'] = df['RP'].astype(int)
    df_extents = df[df['Flood_Extent'].notna()]
    pivot = df_extents.pivot_table(index=['Area', 'RP'], columns='Flood_Extent', values='Area_sqkm', aggfunc='first').fillna(0)
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_percent = pivot_percent.sort_index(level='RP')
    return pivot_percent, pivot

past_percent, past_area = preprocess_df(past_df)
fut_percent, fut_area = preprocess_df(fut_df)

########## Plotting the relative contribution ################
# Define order
ordered_areas = ['Domain', 'LowerPeeDee', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']
return_periods = sorted(past_percent.index.get_level_values('RP').unique())

# Reindex for order and alignment
def reindex_df(df):
    return df.loc[df.index.get_level_values('Area').isin(ordered_areas)].reindex(
        pd.MultiIndex.from_product([ordered_areas, return_periods], names=['Area', 'RP'])
    )

past_percent = reindex_df(past_percent)
fut_percent = reindex_df(fut_percent)

# Plotting
nrows = len(ordered_areas)
fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(6.5, 1.5 * nrows), sharex=True, sharey=True)

colors = {
    'Coastal': 'lightblue',
    'Compound': 'orange',
    'Runoff': 'lightseagreen'
}

for i, area in enumerate(ordered_areas):
    for j, (data, label) in enumerate(zip([past_percent, fut_percent], ['Past', 'Future'])):
        ax = axs[i, j]
        area_df = data.loc[area]
        bars = area_df.plot(kind='bar',
                            stacked=True,
                            ax=ax,
                            color=colors,
                            width=0.6,
                            legend=False)
        ax.set_xticklabels(area_df.index.astype(str), rotation=0)
        if i == 0:
            ax.set_title(label, fontsize=11)

        if j == 0:
            ax.set_ylabel(f'{area}\n\nOverland Extent', fontsize=10)
        else:
            ax.set_ylabel('')

        # Annotate bars
        for container in bars.containers:
            for bar in container:
                height = bar.get_height()
                if height > 3:  # Only label if visible enough
                    ax.annotate(f'{height:.0f}%',
                                xy=(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2),
                                ha='center', va='center', fontsize=9)

        ax.set_xlabel('Return Period')

# Add shared legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', ncol=1,bbox_to_anchor=(1.2, 0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.5, w_pad=0.5)
plt.savefig('relContr_AEP_past_vs_future.jpg', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


########## Plotting scale factors ######################
past_area['Total'] = past_area.sum(axis=1)
fut_area['Total'] = fut_area.sum(axis=1)

sf_area = ((fut_area - past_area) / past_area)+1
sf_area = sf_area.sort_index(level='RP')

# Define flood extent types
flood_types = ['Coastal', 'Compound', 'Runoff', 'Total']

# Define areas and subplot grid
areas = sf_area.index.get_level_values('Area').unique()
ncols = 2
nrows = (len(areas) + 1) // ncols

# Create subplots
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 2 * nrows),
                        sharex=True, sharey=True)
axs = axs.flatten()

colors = {
    'Coastal': 'lightblue',
    'Compound': 'orange',
    'Runoff': 'lightseagreen',
    'Total': 'gray'
}

# Plot lines for each area
for i, area in enumerate(areas):
    ax = axs[i]
    area_data = sf_area.loc[area]
    ax.axhline(y=1, color='black', linewidth=1.5, linestyle='--')
    for flood_type in flood_types:
        ax.plot(area_data.index, area_data[flood_type], label=flood_type, color=colors[flood_type], marker='o')
    ax.set_xscale('log')
    ax.set_ylim(0,10)
    ax.set_title(area)
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('Flood Extent\nScale Factor')
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'])

    # Grid on y-axis only with those ticks
    ax.grid(axis='y', which='major', linestyle=':', linewidth=0.7)

# Shared legend outside plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.5, w_pad=0.5)
plt.savefig('FldExt_RP_SF_by_watershed_log.jpg', bbox_inches='tight', dpi=300)
plt.show()
plt.close()