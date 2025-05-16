import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
mpl.use('TkAgg')
plt.ion()

# Setup plot format
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True

# Load your data as a geopandas dataframe here
d = gpd.read_file()

# Adjust this for the minimum number of points (buildings flooded) to trigger a hexbin
mincnt = 3
# Adjust this to change the size of the hexbin
gridsize = 700

# Plot
nrow = 1
ncol = 1
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 6))

# Change the colormap and bounds depending on what you want (discrete, continuous)
cmap = mpl.colors.ListedColormap(['#3F5565', '#879BEE', '#DD7596'])
bounds = [0, 2, 4, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# Plot the Hexbin (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hexbin.html)
# Input the x, y coordinates of the points and the variable to color them by (depth, category, etc.)
hb = ax.hexbin(d['xcoords'], # x coords
               d['ycoords'], # y coords
               C=d['depth'], # variable we are plotting
               mincnt = mincnt,
               #marginals=True,
               gridsize=gridsize,
               cmap=cmap,
               norm=norm,
               reduce_C_function = np.mean, # can be the mean, sum, max of the points
               #extent = region.total_bounds,  # geopandas dataframe of shapefile of the model region
               alpha=1,
               edgecolors='black',
               linewidth=0.1,
               zorder=1
               )

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(f'hexbin_mincnt{mincnt}_gridsize{gridsize}.jpg', dpi=300, bbox_inches='tight')
plt.close()