import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random


def weighted_sampling_without_replacement(data, weights, num_samples):
    if num_samples > len(data):
        raise ValueError("Number of samples cannot be greater than the number of data points.")

    # Create a list of indices based on the data
    population = list(enumerate(zip(data, weights)))  # each item is now (index, (data, weight))
    sampled_indices = []

    for _ in range(num_samples):
        total_weight = sum(w for _, (_, w) in population)
        selected_index = random.choices(
            range(len(population)), weights=[w for _, (_, w) in population], k=1
        )[0]

        sampled_item, _ = population.pop(selected_index)  # Remove the item
        sampled_indices.append(sampled_item)  # Store the index

    return sampled_indices


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\BiasCorrection')
gcms = ['canesm', 'cnrm6', 'ecearth6', 'ipsl6', 'miroc6']

# Combine all the TC tracks and their weights into a single dataframe
full_set = pd.DataFrame()
for gcm in gcms:
    df = pd.read_csv(f'{gcm}_ssp585_weighted.csv', header=None)
    df.columns = ['tc_id','vmax', 'gcm_weight']
    df['vmax'] = df['vmax'].astype(float)
    df['gcm_weight'] = df['gcm_weight'].astype(float)
    df['gcm_tcid'] = [f'{gcm}_{x}' for x in df['tc_id']]
    full_set = pd.concat(objs=[full_set,df], axis = 0, ignore_index=True)

    sorted_data = df.sort_values(by='vmax', axis=0, ascending=True)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    #Plot the Empirical CDF
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True, sharey=True)
    ax.plot(sorted_data['vmax'], cdf,
            linestyle='-', label='Raw Full Set', alpha=0.7, color='red')
    ax.plot(sorted_data['vmax'], sorted_data['gcm_weight'].cumsum(),
            linestyle='-', label=f'Bias Corrected', alpha=0.7, color='blue')
    ax.set_xlabel('Vmax', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.legend()
    ax.set_title(f'{gcm}')
    ax.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0,0)
    plt.savefig(fr'track_vmax_cdf_{gcm}.png', dpi=300, bbox_inches="tight")
    plt.close()


# Normalize the weights (sum should be 1)
full_set['full_set_weight'] = full_set['gcm_weight'].to_numpy()/full_set['gcm_weight'].cumsum().values[-1]
full_set_ordered = full_set.sort_values(by='vmax',axis=0, ascending=True)
full_set_ordered.reset_index(inplace=True)
full_set_ordered.drop(columns='index', inplace=True)
#full_set_ordered.to_csv('full_set.csv', index=False)
print(full_set_ordered['full_set_weight'].sum())
print(full_set_ordered['full_set_weight'].cumsum().values[-1])


# Weighted sampling without replacement
# Select the number of storms you want to sample
for num_samples in np.arange(5000, 9500, 500):
    # Randomly sample the storms based on their weight and without replacement (e.g., no duplicate storms sampled)
    # selected = np.random.choice(len(full_set_ordered.index),
    #                             size=num_samples,
    #                             replace=False,
    #                             p=full_set_ordered['full_set_weight'].to_numpy())
    #
    # rng = np.random.default_rng()
    # selected2 = rng.choice(len(full_set_ordered.index), size=num_samples, replace=False, p=full_set_ordered['full_set_weight'].to_numpy())
    selected3 = weighted_sampling_without_replacement(data=full_set_ordered['vmax'].to_numpy(),
                                                weights=full_set_ordered['full_set_weight'].to_numpy(),
                                                num_samples=num_samples)
    subset = full_set_ordered[full_set_ordered.index.isin(selected3)].copy()
    subset_ordered = subset.sort_values(by='vmax', axis=0, ascending=True)
    # Normalize the weights for the sample tracks
    subset_ordered['sample_set_weight'] = subset_ordered['full_set_weight'].to_numpy() / subset_ordered['full_set_weight'].cumsum().values[-1]
    subset_ordered.to_csv(f'samples_set_{num_samples}.csv', index=False)
    print(subset_ordered['sample_set_weight'].sum())
    print(subset_ordered['sample_set_weight'].cumsum().values[-1])

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(full_set_ordered['vmax'], subset_ordered['vmax'])

    cdf = np.arange(1, len(full_set_ordered) + 1) / len(full_set_ordered)

    #Plot the Empirical CDF
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True, sharey=True)
    ax.plot(full_set_ordered['vmax'].to_numpy(), cdf,
            linestyle='-', label=f'Raw Full Set ({len(full_set_ordered)})', alpha=0.7, color='red')
    ax.plot(full_set_ordered['vmax'].to_numpy(), full_set_ordered['full_set_weight'].cumsum().to_numpy(),
            linestyle='-', label=f'Bias Corrected Full Set ({len(full_set_ordered)})', alpha=0.7, color='green')
    ax.plot(subset_ordered['vmax'].to_numpy(), subset_ordered['sample_set_weight'].cumsum().to_numpy(),
            linestyle='-', label=f'Bias Corrected Sample ({num_samples})', alpha=0.7, color='blue')
    ax.set_xlabel('Vmax', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.legend()
    ax.set_title(f'ks_stat:{np.round(ks_statistic,3)}\np-value:{np.format_float_scientific(p_value, 3)}')
    ax.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0,0)
    plt.savefig(fr'track_vmax_cdf_{num_samples}_v3.png', dpi=300, bbox_inches="tight")
    plt.close()

