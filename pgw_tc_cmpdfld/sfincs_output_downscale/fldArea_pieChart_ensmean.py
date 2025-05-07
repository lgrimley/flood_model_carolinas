import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
mpl.use('TkAgg')
plt.ion()
import pandas as pd
import numpy as np
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\downscale_test')
df = pd.read_csv(r'downscale_comparsion_depth_fldArea.csv')
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df['attrCode'] = df['attrCode'].map(lambda x: mapping.get(x, 'Total'))

df = df[['storm', 'climate', 'attrCode', 'Area_sqkm', 'gridRes']]
df['group'] = df['storm'].astype(str) + '_' + df['climate'].astype(str)
df.set_index('group', inplace=True, drop=True)


res = 5

dfs = df[df['gridRes'] == res].copy()
dfs.drop(columns=['storm', 'climate', 'gridRes'], inplace=True)
df_reset = dfs.reset_index()  # bring 'group' back as a column to pivot
df_pivoted = df_reset.pivot_table(index='group', columns='attrCode', values='Area_sqkm', aggfunc='first')
df_pivoted.drop(columns=['Total'], inplace=True)
pie_scale = df_pivoted.sum(axis=1) / 14000

# Plotting
runs_to_plot = ['flor_present', 'flor_future', 'floy_present', 'floy_future', 'matt_present', 'matt_future']
storms = ['Florence', 'Floyd', 'Matthew']

# Setup Plot Info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
colors = ['#4F6272', '#DD7596', '#B7C3F3']
nrow, ncol = 3, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 5.5))
axs = axs.flatten()
for i in range(len(runs_to_plot)):
    ax = axs[i]
    d = df_pivoted[df_pivoted.index == runs_to_plot[i]]
    ax.pie(d.to_numpy()[0],
           colors=colors,
           radius=pie_scale[pie_scale.index == runs_to_plot[i]][0],
           startangle=90,
           autopct='%1.1f%%'
           )
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.1, 0.5, storms[i],
                              horizontalalignment='right', verticalalignment='center',
                              rotation='horizontal',transform=axs[first_in_row[i]].transAxes)
axs[0].set_title('Present')
axs[1].set_title('Future')
legend_kwargs0 = dict(bbox_to_anchor=(2.2, 0.8), title=None, loc="upper right", frameon=True, prop=dict(size=10))
axs[3].legend(labels=df_pivoted.columns, **legend_kwargs0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
#plt.tight_layout()
fig.suptitle(f'{res}m')
plt.savefig(f'fld_area_pieChart_sbgRes{res}.jpg', bbox_inches='tight', dpi=300)
plt.close()
