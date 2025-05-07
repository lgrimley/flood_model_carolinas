import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from vinecopulas.marginals import *
from vinecopulas.bivariate import *
#from vinecopulas.vinecopula import *
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True

# Working directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob')
data_files = os.listdir()
file = r'.\basin_data\Neuse_data_rp_future.csv'
data = pd.read_csv(file).astype(float)
data.rename(columns={'Unnamed: 0': 'tc_id'}, inplace=True)
print(data.head())
# ['tc_id', 'maxWS', 'meanMaxWS', 'meanWS', 'meanWSthresh',
#        'meanDirection', 'CumPrecipKM3', 'MeanTotPrecipMM', 'MaxTotPrecipMM',
#        'maxRR', 'meanRR', 'meanRRthresh', 'stormtide', 'FldArea', 'vmax',
#        'weight', 'maxWS_rp', 'meanMaxWS_rp', 'meanWS_rp', 'meanWSthresh_rp',
#        'meanDirection_rp', 'CumPrecipKM3_rp', 'MeanTotPrecipMM_rp',
#        'MaxTotPrecipMM_rp', 'maxRR_rp', 'meanRR_rp', 'meanRRthresh_rp',
#        'stormtide_rp', 'FldArea_rp', 'basin', 'period'],

os.chdir(r'.\copula')

# Load in the two data variables to fit a copula to
df = data[['maxWS', 'MaxTotPrecipMM']]

# turn dataframe into an array
x = np.array(df)

# We need to transform the data into uniform margins (pseudodata)
u = pseudodata(x)
u1 = u[:,0]
u2 = u[:,1]

# Plot the raw data and transformed data
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].scatter(df.maxWS, df.MaxTotPrecipMM, marker='.', label="original data", color='#1f77b4')
axes[0].set_xlabel('Max WndSpd (m/s)')
axes[0].set_ylabel('Max Tot Precip (mm)')
axes[1].scatter(u1, u2, marker='.', label="transformed data", color='green')
axes[1].set_xlabel('u1')
axes[1].set_ylabel('u2')
plt.tight_layout()
plt.savefig('original_and_transformed_data.png', dpi=300)
plt.close()

# Fit a copula to transformed data
# We will try fitting all copulas
cops = list(range(1,10))
# Return the best fit copula with the lowest AIC
cop, par, AIC = bestcop(cops, u)
print(f'Best fit copula: {copulas[cop]}')
# The Frank copula is particularly known for its ability to model positive dependence between random variables,
# meaning it can capture situations where the two variables are likely to increase or decrease together.

# Calculate the CDF of the copula
y = CDF(cop, u, par) #CDF

# Plot the CDF of the copula
U1, U2 =  np.meshgrid(u1, u2)
x1 = 1 - np.arange(0,1,0.01)
y1 =  1 - np.arange(0,1,0.01)
X,Y = np.meshgrid(x1,y1)
z = CDF(cop, np.vstack((X.flatten(), Y.flatten())).T, par)
Z = np.resize(z,X.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=10, cstride=10, linewidth=1)
ax.set_xlabel('$u_1$')
ax.set_ylabel('$u_2$')
ax.set_zlabel('CDF', rotation=90)
ax.zaxis.labelpad=-0.7
plt.title(copulas[cop])
plt.tight_layout()
plt.savefig(f'CDF_for_{copulas[cop]}.png', dpi=300)
plt.close()

# Calculate the PDF of the best fit copula
# This describes the density of probability at a particular point in the joint distribution space defined by the copula.
# It quantifies the likelihood of observing a specific combination of quantiles for the variables.
p = PDF(cop, u, par) #PDF

# Conditional Cumulative Distribution Function (h-function)
v2 = hfunc(cop, u1, u2, par, un = 1) #conditional CDF of Max TP (u2) given Max WS (u1)
v1 = hfunc(cop, u1, u2, par, un = 2) #conditional CDF of Max WS (u1) given Max TP (u2)

# Inverse Conditional Cumulative Distribution Function (inverse h-function)
u2_2 =  hfuncinverse(cop, u1, v2, par, un = 1)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.scatter(x=np.array(u2),y=np.array(u2_2))
plt.xlabel('$u_2$')
plt.ylabel('${C_{1|2}^{-1}(u_1,v_2)}$')
plt.show()
plt.close()

#generate random samples from the conditional copula
n = 10000#len(u)
ur = random(cop, par, n)
plt.scatter(u1,u2, label = 'Data')
plt.scatter(ur[:,0], ur[:,1], alpha = 0.5, label = 'Random')
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.legend()
plt.tight_layout()
plt.savefig(f'Sampled_data_from_copula_{copulas[cop]}.png', dpi=300)
plt.close()


#Fit marginal distributions
x1dist = best_fit_distribution(x[:,0])
x2dist = best_fit_distribution(x[:,1])

# Calculate the CDF of the best fit distribution
u = x.copy()
u[:,0] = x1dist[0].cdf(u[:,0] , *x1dist[1])
u[:,1] = x2dist[0].cdf(u[:,1] , *x2dist[1])

# Transform the simulated uniform data points to resemble the original data using inverse CDF applied to the psuedo data
# calculate the ppf (percent point function) of the previously generated random samples of u1
x1i = x1dist[0].ppf(ur[:,0] , *x1dist[1])
x2i = x2dist[0].ppf(ur[:,1] , *x2dist[1])
plt.scatter(x[:,0],x[:,1], label = 'Data')
#plt.scatter(x1i,x2i, alpha = 0.5, label = 'Random')
plt.xlabel('Max WS')
plt.ylabel('Max TP')
plt.legend()
plt.show()







# corr_sel_MM = sp.kendalltau(sel.loc[:,var2_name].values, sel.loc[:,var1_name].values, nan_policy='omit')
# https://github.com/couasnonanais/seasonality_risk/blob/main/MultivariateDistribution/Variable_monthly_dist_final.py
#