import datetime
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import pymannkendall as mk
import sklearn
import sklearn.linear_model
import datetime

warnings.filterwarnings("ignore")


def get_isu_atm(id, begin_date, end_date):
    """Retrieve atmospheric pressure data from the ISU ASOS download service
    Args:
        id (str): Station id
        begin_date (str): Beginning date of requested time period. Format: %Y%m%d %H:%M
        end_date (str): End date of requested time period. Format: %Y%m%d %H:%M
    """
    print(inspect.stack()[0][3])  # print the name of the function we just entered

    new_begin_date = pd.to_datetime(begin_date, utc=True)
    new_end_date = pd.to_datetime(end_date, utc=True) + timedelta(days=1)

    query = {'station': str(id),
             'data': 'all',
             'year1': new_begin_date.year,
             'month1': new_begin_date.month,
             'day1': new_begin_date.day,
             'year2': new_end_date.year,
             'month2': new_end_date.month,
             'day2': new_end_date.day,
             'product': 'air_pressure',
             'format': 'comma',
             'latlon': 'yes'
             }

    r = requests.get(url='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py', params=query, headers={
        'User-Agent': 'Sunny_Day_Flooding_project, https://github.com/sunny-day-flooding-project'})

    s = slicer(str(r.content, 'utf-8'), "station")
    data = StringIO(s)

    r_df = pd.read_csv(filepath_or_buffer=data, lineterminator="\n", na_values=["", "NA", "M"])

    r_df["date"] = pd.to_datetime(r_df["valid"], utc=True);
    r_df["id"] = str(id);
    r_df["notes"] = "ISU";
    r_df["pressure_mb"] = r_df["alti"] * 1000 * 0.0338639

    r_df = r_df.loc[:, ["id", "date", "pressure_mb", "notes"]].rename(
        columns={"id": "id", "t": "date", "v": "pressure_mb"})

    return r_df


''' Loading and cleaning data '''
os.chdir(r'C:\Users\lelise\University of North Carolina at Chapel Hill\SunnyD Flood Sensor Network - General\Flood '
         r'Protocol\New Bern\Data Analysis\HOBO Comparison')
hobo_file = 'NewBern_NB01_062922_021323.csv'
hobo = pd.read_csv(hobo_file, skiprows=1, infer_datetime_format=True)
dt_utc = pd.to_datetime(hobo.iloc[:, 1]) + datetime.timedelta(hours=-4)  # Convert from EDT to UTC
hobo['datetime'] = pd.to_datetime(dt_utc)
hobo.set_index(keys='datetime', drop=True, inplace=True)  # set datetime UTC as index
hobo.drop(columns=[hobo.columns[0], hobo.columns[1], hobo.columns[4], hobo.columns[5]], inplace=True, axis=0)
hobo.columns = ['P_abs_psi', 'Temp_F']
hobo.dropna(axis=0, inplace=True)

# Load SuDs data
suds = pd.read_csv(
    r'C:\Users\lelise\University of North Carolina at Chapel Hill\SunnyD Flood Sensor Network - General\Flood '
    r'Protocol\New Bern\Data Analysis\NB_2022\data_for_display_NB_2022.csv')
suds['datetime_est'] = pd.to_datetime(suds['date'])  # Datetime is eastern
suds['datetime_utc'] = suds['datetime_est'] + pd.to_timedelta(-5, unit='h')  # convert to UTC
suds.set_index('datetime_utc', inplace=True, drop=False)

# Load atm pressure (NOT the correct suds water level in this file)
atmp = pd.read_csv('NB_01_2022_data.csv', infer_datetime_format=True)
atmp['datetime_est'] = pd.to_datetime(atmp['date'])
atmp['datetime_utc'] = atmp['datetime_est'] + pd.to_timedelta(-5, unit='h')
atmp.set_index(keys='datetime_utc', drop=True, inplace=True)
atmp['atm_p_psi'] = atmp['atm_pressure'] * 0.0145  # 1 hPa (millibar) = 0.0145 PSI
atmp['sensor_p_psi'] = atmp['sensor_pressure'] * 0.0145

''' Timezone and data check w/ plots'''
# Make sure timezones make sense
# tstart = '2022-07-24 09:00:00'
# tend = '2022-07-24 16:00:00'
# df_sensor = suds[(suds.index < tend) & (suds.index > tstart)]
# df_atm = atmp[(atmp.index < tend) & (atmp.index > tstart)]
# df_hobo = hobo[(hobo.index < tend) & (hobo.index > tstart)]

# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
# ax2 = ax.twinx()
# ax.plot(df_sensor.index, df_sensor['sensor_water_level_adj'], color='black')
# ax2.plot(df_atm.index, df_atm['sensor_pressure'], color='orange')
# plt.gcf().autofmt_xdate()
# plt.tight_layout()
# plt.close()
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
# ax2 = ax.twinx()
# ax.plot(df_sensor.index, df_sensor['sensor_water_level_adj'], color='black')
# ax2.plot(hobo.index, hobo['P_abs_psi'], color='orange')
# plt.gcf().autofmt_xdate()
# plt.tight_layout()
# plt.close()

# ALL DATA IS NOW IN UTC
# Create figure and axis objects with subplots()

# Plot the atmospheric pressure compared to HOBO absolute pressure
fig, ax = plt.subplots(2, sharex=True, figsize=(6, 6.5))
plt.rcParams['axes.grid'] = False
ax[0].scatter(atmp.index, atmp['atm_p_psi'], color="green", marker='o', s=10, alpha=0.7, edgecolor='None', zorder=1)
ax[0].set_ylabel("ISU Atmospheric Pressure (PSI)", fontsize=10)
ax[1].scatter(hobo.index, hobo['P_abs_psi'], color="red", marker='o', s=10, alpha=0.7, edgecolor='None', zorder=1)
ax[1].set_ylabel("HOBO Absolute Pressure (PSI)", fontsize=10)
plt.xlim(min(hobo.index), max(hobo.index))
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()
plt.margins(x=0, y=0)
plt.savefig('NB_01 HOBO Pressure vs Atm Pressure (UTC)')
plt.close()

''' Remove linear trend from HOBO data'''
# Remove increasing linear trend from HOBO
mkt = mk.original_test(hobo['P_abs_psi'])
print(mkt)
X = [i for i in range(0, len(hobo))]
X = np.reshape(X, (len(X), 1))
y = hobo['P_abs_psi'].values
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
trend = model.predict(X)
hobo['P_abs_psi_detrend'] = hobo['P_abs_psi'] - hobo['P_abs_psi'] * model.coef_

# Plot HOBO pressure data w/ trendline
fig, ax = plt.subplots(1, figsize=(5, 3))
ax.grid(False)
ax.plot(y)
ax.plot(trend)
plt.ylabel('HOBO Absolute Pressure (PSI)')
plt.xlabel('Time')
plt.tight_layout()
plt.margins(x=0, y=0)
plt.show()
plt.savefig('NB_01 HOBO Abs pressure with trend line')
plt.close()

''' Subset/Combine data '''
t1 = hobo.index.min()
t2 = hobo.index.max()
atm_subset = atmp[(atmp.index <= t2) & (atmp.index >= t1)]
atm_subset = atm_subset.resample('10min').mean()
suds_subset = suds[(suds.index <= t2) & (suds.index >= t1)]
suds_subset = suds_subset.resample('10min').mean()
hobo_subset = hobo.resample('10min').mean()

combined = pd.merge(left=suds_subset, right=atm_subset[['atm_p_psi']], how='outer', right_index=True, left_index=True)
combined = pd.merge(left=combined, right=hobo_subset[['P_abs_psi_detrend']], how='outer', right_index=True, left_index=True)

joined = combined[['atm_p_psi', 'P_abs_psi_detrend', 'sensor_water_level_adj','sensor_elevation','road_elevation']]
joined.dropna(how='any', inplace=True, axis=0)

# P_gauge = P_raw / (rho*g) - P_atm / (rho*g) [units of meters]
psi_2_mh20 = 6894.76  # convert PSI to N/m2
joined['p_gauge_m'] = ((joined['P_abs_psi_detrend'] * psi_2_mh20) / (1000 * 9.81)) - (
        (joined['atm_p_psi'] * psi_2_mh20) / (1000 * 9.81))

joined['sensor_elevation'] = 1.89
joined['road_elevation'] = 4.66

# Start working in FEET NAVD88
joined['hobo_wse_FTnavd88'] = (joined['p_gauge_m'] * 3.28084) + joined['sensor_elevation']

#########################

fig, ax = plt.subplots(figsize=(12, 6))
plt.rcParams['axes.grid'] = False
ax.plot(joined.index, joined['sensor_elevation'], color="black", alpha=0.75, linewidth=1.5, label='Sensor Elevation')
ax.plot(joined.index, joined['road_elevation'], color="orange", linewidth=1.5, label='Road Elevation')
ax.scatter(joined.index, joined['sensor_water_level_adj'],
           color="blue", alpha=0.5, s=20, marker='o', label='SuDS Adjusted WL')
ax.plot(joined.index, joined['sensor_water_level_adj'], color="blue", alpha=0.5, linewidth=0.5)

ax.scatter(joined.index, joined['hobo_wse_FTnavd88'],
           color="red", alpha=0.5, s=20, marker='o', label='HOBO Detrended')
ax.plot(joined.index, joined['hobo_wse_FTnavd88'], color="red", alpha=0.5, linewidth=0.5)

ax.set_ylabel("Water Level (ft +NAVD88)", fontsize=12)
plt.ylim(1.5, 5)
plt.legend(loc='upper right')
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.margins(x=0, y=0)
plt.show()
plt.savefig('NB_01 HOBO and SuDS Water Level with trend removed')
plt.close()

t1f = '2022-11-10 00:00:00'
t2f = '2022-11-20 00:00:00'
t1 = pd.to_datetime(t1f)
t2 = pd.to_datetime(t2f)
joined_sub = joined[(joined.index <= t2) & (joined.index >= t1)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(joined_sub.index, joined_sub['sensor_elevation'], color="black", alpha=0.75, linewidth=1.5, label='Sensor Elevation')
ax.plot(joined_sub.index, joined_sub['road_elevation'], color="orange", linewidth=1.5, label='Road Elevation')
ax.scatter(joined_sub.index, joined_sub['sensor_water_level_adj'],
           color="blue", alpha=0.5, s=30, marker='o', label='SuDS Adjusted WL')
ax.plot(joined_sub.index, joined_sub['sensor_water_level_adj'], color="blue", alpha=0.5, linewidth=1)

ax.scatter(joined_sub.index, joined_sub['hobo_wse_FTnavd88'],
           color="red", alpha=0.5, s=30, marker='o', label='HOBO Detrended')
ax.plot(joined_sub.index, joined_sub['hobo_wse_FTnavd88'], color="red", alpha=0.5, linewidth=1)

ax.set_ylabel("Water Level (ft +NAVD88)", fontsize=12)
plt.ylim(1.5, 5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.margins(x=0, y=0)
plt.show()
plt.gcf().autofmt_xdate()
figname = "hobo_wl_comparsion_zoom5"
plt.savefig(figname)
plt.close()
