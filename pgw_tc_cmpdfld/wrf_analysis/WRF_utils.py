import numpy as np
import xarray as xr
import scipy.stats as ss
from scipy.optimize import curve_fit
import hydromt
import hydromt_sfincs
import pandas as pd


def calc_windspd(da):
    wind_u = da['wind_u']
    wind_v = da['wind_v']
    wind_spd = ((wind_u ** 2) + (wind_v ** 2)) ** 0.5
    da['wind_spd'] = wind_spd
    return da


def subset_data_by_thresholds(data, min_threshold, max_threshold):
    # Subset data
    data = xr.where(cond=(data > min_threshold), x=data, y=np.nan)
    data = xr.where(cond=(data < max_threshold), x=data, y=np.nan)
    return data


def create_raster_mask_with_polygon(data, poly):
    poly['mask'] = 1.0
    mask = data.raster.rasterize(poly, "mask", nodata=np.nan, all_touched=False)
    mask = xr.where(mask == 1.0, x=False, y=True)
    return mask


def expon_func(x, a):
    return a * np.exp(-a * x)


def fit_expon_dist(data, lower_threshold, upper_threshold, nbins):
    # Fit exponential distribution and save params
    loc, scale = ss.expon.fit(data, floc=0)
    counts, bins = np.histogram(data, bins=nbins, range=[lower_threshold, upper_threshold], density=True)
    bin_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    popt, pcov = curve_fit(expon_func, xdata=bin_centers, ydata=counts)
    return loc, scale, popt, pcov

