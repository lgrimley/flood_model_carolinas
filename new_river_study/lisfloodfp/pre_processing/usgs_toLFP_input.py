#!/usr/bin/env python3
# coding: utf-8
########################################################################################################
#    Author: Lauren E. Grimley
#    Contact: lgrimley@unc.edu
########################################################################################################
import pandas as pd
import datetime as dt
import dataretrieval.nwis as nwis
from pytz import timezone


def simulation_time_seconds(data):
    holder = [0]
    for i in range(len(data.index) - 1):
        tdiff = data.index[i + 1] - data.index[i]
        tsec = tdiff.total_seconds()
        tcum = holder[i] + tsec
        holder.append(tcum)
    data['seconds'] = holder

    # Simulation time in seconds
    simtime = data.index[-1] - data.index[0]
    print(simtime)
    print(simtime.total_seconds())
    return data


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


# Pull data for the following datetimes
tstart_utc = '2019-08-18'
tend_utc = '2019-09-14'
model_run = 'New River Dorian'
bdy_name = 'gum_branch_usgs_wl'
file_out = 'C:/Users/lelise/NewRiver_Local/lisflood/dorian/' + bdy_name + '.bdy'

# ------------- Gum Branch USGS ------------------
gb_usgs = nwis.get_record(sites='02093000', service='iv', start=tstart_utc, end=tend_utc, parameterCd='00065')
gb_usgs['00065'] = gb_usgs['00065'] * 0.3048
gb_out = simulation_time_seconds(gb_usgs)

# Output data as .bdy filetype for input into Lisflood-fp
gb_out = gb_usgs[['00065', 'seconds']]
gb_out = gb_out.round({'00065': 3, 'seconds': 0})
pd.DataFrame.to_csv(gb_out, file_out, sep='\t', index=False, header=False)
line_prepender(file_out, line=str(len(gb_out['seconds'])) + '\t' + 'seconds')
line_prepender(file_out, bdy_name)
line_prepender(file_out, model_run)


# ------------- Old Jacksonville Bridge USGS ------------------
ojb_usgs = nwis.get_record(sites='0209303201', service='iv', start=tstart_utc, end=tend_utc, parameterCd='62620')
ojb_usgs['62620'] = ojb_usgs['62620'] * 0.3048
ojb_out = simulation_time_seconds(ojb_usgs)
ojb_out = ojb_usgs[['62620', 'seconds']]
ojb_out = ojb_out.round({'62620': 3, 'seconds': 0})
pd.DataFrame.to_csv(ojb_out,
                    '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/Research/NewRiver/florence/usgs_ojb.bdy',
                    sep='\t',
                    index=False, header=None)


# ------------- Radar Rainfall ------------------
model_run = 'New River Dorian - MRMS'
file_out = 'C:/Users/lelise/NewRiver_Local/lisflood/dorian/dorian_mrms.rain'

rf = 'C:\\Users\\lelise\\NewRiver_Local\\rainfall\\mrms\\Dorian_GRIB2_files\\Dorian_mrms_NewRiver_BasinAvg.txt'
d = pd.read_csv(rf, sep='\t', header=0)
d['datetime'] = pd.to_datetime(d['datetime'], format='%m-%d-%Y %H:%M', utc=True)
d = d[d['datetime'] >= gb_out.index[0]]
d.set_index('datetime', inplace=True)
d['val'] = round(d['val'], 3)


rain = simulation_time_seconds(d)
pd.DataFrame.to_csv(rain, file_out, sep='\t', index=False, header=False)
line_prepender(file_out, line=str(len(d['seconds'])) + '\t' + 'seconds')
line_prepender(file_out, model_run)