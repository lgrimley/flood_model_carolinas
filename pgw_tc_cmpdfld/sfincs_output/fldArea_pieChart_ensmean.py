import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')
from pgw_utils import *



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3'
         r'\process_attribution_mask')

# PRESENT
df = pd.read_csv(r'.\fldArea_by_process.csv', index_col=0)
fld_area_df = cleanup_flood_area_dataframe(df)
fld_area_present = fld_area_df[fld_area_df['climate'] == 'pres']
fld_area_present.drop(columns=['storm', 'climate', 'group', 'Total'], inplace=True)

# Load the ensemble mean calculation
df = pd.read_csv(r'..\ensemble_mean_mask\ensmean_mean_fldArea_by_process.csv', index_col=0)
fld_area_fut_ensmean = cleanup_flood_area_dataframe(df)
fld_area_fut_ensmean.drop(columns=['storm', 'climate', 'group', 'Total'], inplace=True)

combined = pd.concat([fld_area_present, fld_area_fut_ensmean], axis=0)

# Plotting
runs_to_plot = ['flor_pres', 'flor_fut_ensmean',
                'floy_pres', 'floy_fut_ensmean',
                'matt_pres', 'matt_fut_ensmean']
pie_scale = combined.sum(axis=1) / 60000
storms = ['Florence', 'Floyd', 'Matthew']

# Setup Plot Info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
nrow, ncol = 3, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(4, 4),
                        tight_layout=True,
                        layout='constrained'
                        )
axs = axs.flatten()
for i in range(len(runs_to_plot)):
    ax = axs[i]
    d = combined[combined.index == runs_to_plot[i]]
    ax.pie(d.to_numpy()[0],
           colors=colors,
           radius=pie_scale[pie_scale.index == runs_to_plot[i]][0],
           startangle=90
           )
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.05, 0.5, storms[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='horizontal',
                              transform=axs[first_in_row[i]].transAxes)
axs[0].set_title('Present')
axs[1].set_title('Future')
legend_kwargs0 = dict(
    bbox_to_anchor=(2.2, 0.8),
    title=None,
    loc="upper right",
    frameon=True,
    prop=dict(size=10),
)
axs[3].legend(labels=combined.columns, **legend_kwargs0)
plt.subplots_adjust(wspace=0.0, hspace=0.05)
plt.margins(x=0, y=0)
plt.savefig('fld_area_pieChart_ensmean.jpg', bbox_inches='tight', dpi=300)
plt.close()
