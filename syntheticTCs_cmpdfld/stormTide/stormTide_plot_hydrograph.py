import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\stormTide\NCEP')
gp_df = pd.read_csv('gage_peaks.csv', index_col=0)

# Plot a histogram of the number of storms with various number of WL gages
plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(figsize=(5, 7),
                        tight_layout=True,
                        nrows=6,ncols=4,
                        sharex=True, sharey=True)
axs = axs.flatten()
axs[-1].axis('off')
axs[-2].axis('off')
for i in range(len(gp_df.columns)):
    ax = axs[i]
    gd = gp_df.iloc[:,i]
    gd.hist(ax=ax, bins=20, alpha=0.9, grid=False)
    ax.set_xlim(0, 4)
    ax.set_xticks(ticks=np.arange(0, 5, 1))
    ax.set_title(f'Gage: {gp_df.columns[i]}\n(n={len(gd[~gd.isna()])})', fontsize=8)
plt.xlabel('ADCIRC Peak Surge (m+MSL)')
plt.ylabel('Frequency')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('gage_peak_wl_histogram.png')
plt.close()



