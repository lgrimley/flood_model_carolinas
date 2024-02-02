
# Hydrograph Area Calculation
t11 = pd.Timestamp(ts_input=t1, tz='UTC')
t22 = pd.Timestamp(ts_input=t2, tz='UTC')
usgs_mask = (gb_usgs.index > t11) & (gb_usgs.index <= t22)
ojb_usgs_vol = gb_usgs.loc[usgs_mask]
a = []
for i in range(len(ojb_usgs_vol.index) - 1):
    dt = abs((ojb_usgs_vol.index[i] - ojb_usgs_vol.index[i + 1]).total_seconds())
    area = dt * ojb_usgs_vol['00065'][i]
    a.append(area)
ttime = abs((ojb_usgs_vol.index[0] - ojb_usgs_vol.index[len(ojb_usgs_vol.index) - 1]).total_seconds())
usgs_vol = np.sum(a) / ttime

lfp_ojb = d[0].drop(columns=['Station_1', 'Station_3'])
a = []
for i in range(len(lfp_ojb.index) - 1):
    dt = abs((lfp_ojb.index[i] - lfp_ojb.index[i + 1]).total_seconds())
    area = dt * lfp_ojb['Station_2'][i]
    a.append(area)
ttime = abs((lfp_ojb.index[0] - lfp_ojb.index[len(lfp_ojb.index) - 1]).total_seconds())
ojb_vol = np.sum(a) / ttime
diff = ojb_vol - usgs_vol

# STATS
usgs_ojb_rs = ojb_usgs_vol.resample('30min', convention='start').asfreq()
usgs_ojb_rs = usgs_ojb_rs.drop(axis='index', labels=usgs_ojb_rs.index[0])
lfp_ojb_rs = lfp_ojb.resample('30min', convention='start').asfreq()

x1 = usgs_ojb_rs['00065'].to_numpy()
x2 = lfp_ojb_rs['Station_2'].to_numpy()
rmse = np.sqrt(((x1 - x2) ** 2).mean())
mae = abs((x1 - x2)).mean()
mae_check = sum(abs(x1 - x2)) / len(x1)