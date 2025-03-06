import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True


import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True


''' 
Get total buildings flooded for historical storms 
'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')
bld_tot_df_filpath = 'bld_fld_counts_FlorFloyMatt.csv'
bld_tot_df = pd.read_csv(bld_tot_df_filpath, index_col=0)
bld_tot_df['Storm'] = [x.split('_')[0] for x in bld_tot_df.index]
storms = np.unique(bld_tot_df['Storm'])
# Mean Absolute Error depth threshold
subset = bld_tot_df[bld_tot_df['hmin'] == 0.64]
build_storms = subset[['Coastal', 'Runoff', 'Compound', 'Storm', 'Period']]
build_storms['Metric'] = 'Buildings'
build_storms.set_index('Storm', drop=True,inplace=True)

# Pull in the flood extent data
fld_extent = pd.read_csv(r'..\Chapter2_PGW\sfincs\03_OBS\analysis_final\process_attribution_mask\stats_for_manuscript.csv',
                         index_col=0)
fld_extent_storms_pres = fld_extent[fld_extent.index.isin(['flor_pres','floy_pres','matt_pres'])][['Coastal', 'Runoff', 'Compound']]
fld_extent_storms_pres['Storm'] = ['flor', 'floy', 'matt']
fld_extent_storms_pres.set_index('Storm', drop=True,inplace=True)
fld_extent_storms_pres['Period'] = 'Present'
fld_extent_storms_pres['Metric'] = 'Extent'

fld_extent_futmean = pd.read_csv(r'..\Chapter2_PGW\sfincs\03_OBS\analysis_final\ensemble_mean_mask\ensmean_mean_fldArea_by_process.csv',
                                 index_col=0)
fld_extent_storms = fld_extent_futmean.T
fld_extent_storms['Compound'] = fld_extent_storms['compound_coastal'] + fld_extent_storms['compound_runoff']
fld_extent_storms = fld_extent_storms[['coastal','runoff','Compound']]
fld_extent_storms.columns = ['Coastal','Runoff','Compound']
fld_extent_storms['Storm'] = ['flor','matt','floy']
fld_extent_storms.set_index('Storm', drop=True,inplace=True)
fld_extent_storms.sort_index(axis=0, inplace=True)
fld_extent_storms['Period'] = 'Future'
fld_extent_storms['Metric'] = 'Extent'

fld_storms = pd.concat(objs=[fld_extent_storms_pres, fld_extent_storms, build_storms], axis=0, ignore_index=False)

''' 1% storm '''
building_df = pd.read_csv('buildings_rps_exposure.csv', index_col=0, low_memory=True)
bld_tot_df_filpath = 'bld_fld_counts_rps.csv'
bld_tot_df = pd.read_csv(bld_tot_df_filpath, index_col=0)
subset = bld_tot_df[bld_tot_df['hmin'] == 0.64]
d1 = subset[subset['RP']==100][['Coastal', 'Runoff', 'Compound', 'Period']]
d1['Storm'] = '100yr'
d1['Metric'] = 'Buildings'
d1.set_index('Storm', drop=True,inplace=True)


# Flood extent
fld_extentAEP = pd.read_csv(r'..\Chapter3_SyntheticTCs\04_RESULTS\results_attribution\aep_extent_rel_contribution_Wbmasked.csv',
                                 index_col=0)
fld_extentAEP = fld_extentAEP.T
fld_extentAEP = fld_extentAEP[['Coastal','Runoff','Compound']]
fld_extentAEP['RP'] = [int(x.split('_')[-1]) for x in fld_extentAEP.index]
fld_extentAEP['Period'] = [x.split('_')[0] for x in fld_extentAEP.index]
fld_extentAEP = fld_extentAEP[fld_extentAEP['RP'] == 100]
fld_extentAEP['Period'] = ['Present', 'Future']
fld_extentAEP['Metric'] = 'Extent'
fld_extentAEP['Storm'] = '100yr'
fld_extentAEP.set_index('Storm', drop=True,inplace=True)

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

# Plot pie chart
colors = ['#3F5565', '#879BEE', '#DD7596']
combined = pd.concat(objs=[fld_storms, d1, fld_extentAEP], axis=0, ignore_index=False)

storm = '100yr'
s = combined[combined.index == storm]
s.reset_index(inplace=True, drop=True)
new_order = [2,3,1,0]
s = s.reindex(new_order)
s.drop('RP', axis=1, inplace=True)
s = s[['Coastal', 'Runoff', 'Compound']]
s = s.reset_index(drop=True)
sum = s.sum(axis=1)
pie_scale = [71000,71000,36000,36000]
ps = sum.div(pie_scale)

nrow, ncol = 2, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(4, 4),
                        tight_layout=True,
                        layout='constrained'
                        )
axs =axs.flatten()
for i in range(len(s.index)):
    ax = axs[i]
    d = s.iloc[i,:]
    d = d.to_numpy()
    wedges, texts, autotexts = ax.pie(d,
           colors=colors,
           radius=ps[i],
           startangle=90,
           autopct='%1.0f%%',#pctdistance=0.5
           )
    theta = [((w.theta2 + w.theta1) / 2) / 180 * np.pi for w in wedges]
    if i <2:
        adjust_labels(texts, theta, 1.25)
        adjust_labels(autotexts, theta, 0.4)

axs[0].set_title('Present')
axs[1].set_title('Future')
legend_kwargs0 = dict(
    bbox_to_anchor=(0.9, 1.2),
    title=None,
    loc="upper right",
    frameon=True,
    prop=dict(size=10),
)
#axs[4].legend(labels=combined.columns, **legend_kwargs0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig('100yr_pieChart_extent_buildings.png', dpi=300)
plt.close()

