import pandas as pd
import sys
import os
import numpy as np


wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\probabilistic_WSE\floodmaps_5m'
os.chdir(wdir)
csvfiles = [f for f in os.listdir(wdir) if f.endswith('.csv')]
print(csvfiles)

# Present AEP
df1 = pd.read_csv(csvfiles[1], index_col=0)  ## need to select the compound CSV
df1.index = df1.index.str.replace(' ', '', regex=False)
df1 = df1[~df1.index.duplicated(keep='last')]
df1.index = df1.index.str.split('_', expand=True)
df1.index.names = ['rp','basin','attr']
df1.reset_index(inplace=True)
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df1['attr'] = df1['attr'].map(lambda x: mapping.get(x, 'Total'))
df_hist = df1
domain_hist = df_hist[df_hist['basin'] == 'Domain']
domain_hist.reset_index(inplace=True, drop=True)


wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\probabilistic_WSE\floodmaps_5m'
os.chdir(wdir)
csvfiles = [f for f in os.listdir(wdir) if f.endswith('.csv')]
print(csvfiles)

# Future AEP
df1 = pd.read_csv(csvfiles[0], index_col=0)
df1.index = df1.index.str.replace(' ', '', regex=False)
df1 = df1[~df1.index.duplicated(keep='last')]
df1.index = df1.index.str.split('_', expand=True)
df1.index.names = ['rp','basin','attr']
df1.reset_index(inplace=True)
mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
df1['attr'] = df1['attr'].map(lambda x: mapping.get(x, 'Total'))
df_fut = df1
domain_fut = df_fut[df_fut['basin'] == 'Domain']
domain_fut.reset_index(inplace=True, drop=True)

# subtract the historic from the future
numeric_cols = domain_hist.select_dtypes(include='number').columns
diff = (domain_fut[numeric_cols] - domain_hist[numeric_cols])
diff['sf_area'] = 1 + diff['Area_sqkm'].div(domain_hist['Area_sqkm'])
diff = np.round(diff,4)
diff[['rp','basin', 'attr']] = domain_hist[['rp','basin', 'attr']]

# Loop through the return periods and get the relative contribution of the 3 flood processes (as a %)
df = domain_hist[['Area_sqkm',	'rp','attr']]
result_list = [] # Container for results
# Loop through each return period
for rp, group in df.groupby('rp'):
    subset = group.set_index('attr')['Area_sqkm']
    data = subset[['Coastal', 'Compound', 'Runoff']]
    total = data.sum()
    percentages = (data / total) * 100
    result = percentages.reset_index()
    result.columns = ['attr', 'percent']
    result['rp'] = rp
    result_list.append(result)

# Combine all into one DataFrame
result_df = pd.concat(result_list, ignore_index=True)
pivot_df = result_df.pivot(index='rp', columns='attr', values='percent')
print(pivot_df.round(2))

# repeat for the future
df = domain_fut[['Area_sqkm','rp','attr']]
result_list = [] # Container for results
# Loop through each return period
for rp, group in df.groupby('rp'):
    subset = group.set_index('attr')['Area_sqkm']
    data = subset[['Coastal', 'Compound', 'Runoff']]
    total = data.sum()
    percentages = (data / total) * 100
    result = percentages.reset_index()
    result.columns = ['attr', 'percent']
    result['rp'] = rp
    result_list.append(result)

# Combine all into one DataFrame
result_df = pd.concat(result_list, ignore_index=True)
pivot_df = result_df.pivot(index='rp', columns='attr', values='percent')
print(pivot_df.round(2))