library(dataRetrieval)
library(ggplot2)
library(magrittr)

####### Functions #######
compute_seconds <- function(data){
  tsec = c(0)
  for (i in 1:(length(data$dateTime)-1)){
    tdiff = difftime(time1 = data$dateTime[i+1], 
                     time2 = data$dateTime[1],
                     tz = 'UTC',
                     units='secs')
    tsec = append(tsec,tdiff)
  }
  data['seconds']=tsec
  return(data)
}

get_record <-function(siteNumber, parameterCd){
  d = whatNWISdata(siteNumbers = siteNumber, service = 'uv')
  if (parameterCd %in% d$parm_cd){
    ind = which(d$parm_cd == parameterCd)
    for (i in 1:length(ind)){
      x = d[ind[i],]
      if ('wat' %in% x$medium_grp_cd && is.na(x$loc_web_ds)){
        final_ind = ind[i]
        break
      } 
    }
  }
  x = d[final_ind,]
  
  siteInfo = readNWISsite(siteNumber=siteNumber)
  local_tz = siteInfo$tz_cd
  tinit = as.POSIXct(x$begin_date, format = 'Y-%m-%d', tz = local_tz)
  tend = as.POSIXct(x$end_date, format = 'Y-%m-%d', tz = local_tz)
  
  record = c(tinit, tend)
  return(record)
}

get_usgs_ts <- function(siteNumber, parameterCd, startDate, endDate, tz){
  record = get_record(siteNumber = siteNumber,
                      parameterCd = parameterCd)
  
  if (record[1] > as.POSIXct(startDate, format = '%Y-%m-%d', tz)){
    print("Error: Fix input date because gage record start date is")
    print(record[1])
  } else if ((record[2] < as.POSIXct(endDate, format = '%Y-%m-%d', tz))){
    print("Error: Fix input date because gage record end date is")
    print(record[1])
  } else {
    data = readNWISuv(siteNumbers = siteNumber,
                    parameterCd = parameterCd,
                    startDate = startDate,
                    endDate = endDate,
                    tz = tz)
    data = compute_seconds(data)
    return(data) 
    }
}

####### User Input #######
startDate =  '2016-9-28'
endDate = '2016-10-10'

#Simulation dates
tstart = as.POSIXct(startDate,tz='UTC')
tend = as.POSIXct(endDate,tz='UTC')

# USGS site numbers and parameterCd
#gb = c('02093000','00060')
gb = c('02093000','00065')
ojb = c('0209303201','62620')

####### Execute #######
gb.data = get_usgs_ts(siteNumber = gb[1],
                      parameterCd = gb[2],
                      startDate = startDate,
                      endDate = endDate,
                      tz = 'UTC')

ojb.data = get_usgs_ts(siteNumber = ojb[1],
                      parameterCd = ojb[2],
                      startDate = startDate,
                      endDate = endDate,
                      tz = 'UTC')

# Convert from FT to Meters
#gb.data['q_cms'] = round((gb.data$X_00060_00000*0.028316846592)/30, digits = 4)
gb.data['wse_m'] = round(gb.data$X_00065_00000*0.3048, digits = 3)
ojb.data['wse_m'] = round(ojb.data$X_62620_00000*0.3048, digits = 3)

# Check start datetime and simulation duration
print(gb.data$dateTime[1])
simtime1 = difftime(time1 = gb.data$dateTime[1], 
                    time2 = gb.data$dateTime[length(gb.data$dateTime)],
                    tz = 'UTC',
                    units='secs')
print(simtime1)
simtime2 = difftime(time1 = ojb.data$dateTime[1], 
                    time2 = ojb.data$dateTime[length(ojb.data$dateTime)],
                    tz = 'UTC',
                    units='secs')
print(simtime2)

# Write out data formatted as LISFLOOD-FP input
setwd("C:\\Users\\lelise\\NewRiver_Local\\lisflood\\matthew\\input\\")
print(getwd())

write.table(subset(gb.data, select=c("wse_m","seconds")),
            file='usgs_gum_branch_discharge.bdy',
            sep = '\t',row.names = FALSE)

write.table(subset(ojb.data, select=c("wse_m","seconds")),
            file='usgs_ojb.bdy',
            sep = '\t',row.names = FALSE)


######## Rainfall ########
rainfall_file = 'C:\\Users\\lelise\\Research_Local\\rainfall\\mrms\\Matthew_mrms_NewRiver_BasinAvg.txt'
r = read.delim(file=rainfall_file, header=TRUE, sep='\t')
r$datetime = as.POSIXct(r$datetime, format='%m-%d-%Y %H:%M', tz="UTC")
r$val = round(r$val, digits=3)

colnames(r) = c('dateTime','val')
rsub = r[r$dateTime >= tstart,]
rsub = compute_seconds(rsub)

# Write out data formatted as LISFLOOD-FP input
write.table(subset(rsub, select=c("val","seconds")),
            file='C:\\Users\\lelise\\Research_Local\\rainfall\\mrms\\lfp_mrms_matthew.rain',
            sep = '\t',row.names = FALSE)

######## ADCIRC ########
adcirc_file = 'C:\\Users\\lelise\\OneDrive - University of North Carolina at Chapel Hill\\Research\\NewRiver\\data\\water_level_data\\ADCIRC\\MattNewRiv.csv'
adcirc = read.csv(file=adcirc_file, header=TRUE)
adcirc$Datetime = as.POSIXct(adcirc$Datetime, format='%d-%b-%Y %H:%M:%S', tz="UTC")
adcirc = adcirc[,1:2]
colnames(adcirc) = c('dateTime','val')
rsub = adcirc#[adcirc$dateTime >= tstart,]
rsub = compute_seconds(rsub)
rsub$val = rsub$val - 0.3  # MSL to NAVD88

# Write out data formatted as LISFLOOD-FP input
write.table(subset(rsub, select=c("val","seconds")),
            file='C:\\Users\\lelise\\OneDrive - University of North Carolina at Chapel Hill\\Research\\NewRiver\\data\\water_level_data\\ADCIRC\\lfp_adcirc_bc_matthew.bdy',
            sep = '\t',row.names = FALSE)

####### Plotting #######
ggplot(NULL, aes(x=dateTime, y=wse_m))+
  geom_line(data=gb.data, color='black', lwd = 1.5)+
  geom_line(data=ojb.data, color='red', lwd = 1.5)
  xlab('DateTime (UTC)')+ylab('WSE (m NAVD88)')
  
ggplot(rsub, aes(x=dateTime, y=val))+
  geom_line()
