import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')
from pgw_utils import *


# Area stuff
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3'
         r'\process_attribution_mask')

# All runs
df = pd.read_csv(r'.\fldArea_by_process.csv', index_col=0)
df = cleanup_flood_area_dataframe(df)
# Calculate fractional contribution
for scen in ['Coastal', 'Runoff', 'Compound']:
    df[f'{scen}_RelCont'] = df[scen] / df['Total']
#df.to_csv('stats_for_manuscript.csv')


# Get depths and test significance of the change in depth from present and future ensemble mean for each driver
work_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3'
os.chdir(os.path.join(work_dir, 'ensemble_mean_mask'))

yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base])
dep = mod.grid['dep']

zsmax = xr.open_dataarray(os.path.join(work_dir, 'pgw_zsmax.nc'))
zsmax_class = xr.open_dataarray(os.path.join(work_dir, 'process_attribution_mask' ,'processes_classified.nc'))

fut_ensmean_zsmax = xr.open_dataarray(os.path.join(work_dir, 'ensemble_mean', 'fut_ensemble_zsmax_mean.nc'))
fut_ensmean_class = xr.open_dataarray(os.path.join(work_dir, 'ensemble_mean_mask', 'processes_classified_ensmean_mean.nc'))

for storm in ['flor', 'floy', 'matt']:
    for scenario in ['coastal', 'runoff', 'compound']:
        pres_class = zsmax_class.sel(run=f'{storm}_pres')
        fut_class = fut_ensmean_class.sel(run=f'{storm}_fut_ensmean')

        # No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
        if scenario == 'coastal':
            mask_pres = xr.where((pres_class == 1), True, False)
            mask_fut = xr.where((fut_class == 1), True, False)
        elif scenario == 'runoff':
            mask_pres = xr.where((pres_class == 3), True, False)
            mask_fut = xr.where((fut_class == 3), True, False)
        else:
            # Combine into a single 'compound' classification
            mask_pres = xr.where((pres_class == 2) | (pres_class == 4), True, False)
            mask_fut = xr.where((fut_class == 2) | (fut_class == 4), True, False)


        pres_zsmax = zsmax.sel(run=f'{storm}_pres_{scenario}')
        pres_depth = (pres_zsmax.where(mask_pres) - dep).compute()
        pres_depth = pres_depth.where(pres_depth > 0.05)
        pres_depth.name = f'{storm}_pres_{scenario}'

        fut_zsmax = fut_ensmean_zsmax.sel(run=f'{storm}_fut_{scenario}_mean')
        fut_depth = (fut_zsmax.where(mask_fut) - dep).compute()
        fut_depth = fut_depth.where(fut_depth > 0.05)
        fut_depth.name = f'{storm}_fut_{scenario}_mean'

        # Wilcoxon rank-sum statistic for two independent samples that are not normally distributed.
        # A negative statistic (U) indicates that the group1 (present) tends to have lower values compared to the group2 (future)
        # It also suggests that the null hypothesis (no difference between the groups) is likely not true
        # If p-value is less than 0.05, we reject the null hypothesis
        # Group1
        ds = xr.where(pres_depth == 0, np.nan, pres_depth)
        df = ds.to_dataframe().dropna(how='any', axis=0)
        p = df.drop(columns='spatial_ref')
        # Group2
        ds = xr.where(fut_depth == 0, np.nan, fut_depth)
        df = ds.to_dataframe().dropna(how='any', axis=0)
        f = df.drop(columns='spatial_ref')

        group1 = p[pres_depth.name].values.astype(float)
        group2 = f[fut_depth.name].values.astype(float)
        m1 = np.median(group1)
        m2 = np.median(group2)
        print(f'{storm} - {scenario}')
        print(f"Median depth Pres and Fut is {np.round(m1, 2)} and {np.round(m2, 2)}")
        print(f"Mean depth Pres and Fut is {np.round(np.mean(group1), 2)} and {np.round(np.mean(group2), 2)}")

        statistic, p_value = scipy.stats.ranksums(group1, group2, alternative='two-sided')
        print(
            f"Mann–Whitney U = {np.round(statistic, 2)}, nPres = {len(group1)}; nFut = {len(group2)}, P = {p_value}; two-sided.")

        statistic, p_value = scipy.stats.ranksums(group1, group2, alternative='less')
        print(
            f"Mann–Whitney U = {np.round(statistic, 2)}, nPres = {len(group1)}; nFut = {len(group2)}, P = {p_value}; less.")

        statistic, p_value = scipy.stats.ranksums(group1, group2, alternative='greater')
        print(
            f"Mann–Whitney U = {np.round(statistic, 2)}, nPres = {len(group1)}; nFut = {len(group2)}; P = {p_value};\nThis test says that Present is greater than Future. P = 1 says reject this hypothesis.")

