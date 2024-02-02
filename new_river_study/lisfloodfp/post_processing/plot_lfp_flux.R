library(raster)
library(sf)
library(xts)

#################    Functions   #####################
get_filenames <- function(dir, pattern){
  extension = paste0('"*.', pattern)
  files = list.files(path=dir,pattern=extension, full.names=TRUE)
  return(files)
}

create_stations_sf <- function(stations.df, x_col, y_col, epsg){
  stations.sf.point <- st_as_sf(x = stations.df, 
                                coords = c(x_col, y_col),
                                crs = paste0("+init=EPSG:", epsg))
  return(stations.sf.point)
}

get_flux <- function(files, stations.df, epsg){
  stations.sf.point = create_stations_sf(stations.df,
                                         "x_coord", "y_coord", epsg)
  df = data.frame(matrix(ncol=length(stations.df$site_name)))
  colnames(df) = stations.df$site_name
  for (i in 1:length(files)){
    ras <- raster(files[i])
    crs(ras) <- CRS('+init=EPSG:32119')
    df[i,] <- as.numeric(raster::extract(ras, stations.sf.point))
  }
  return(df)
}

##################   Input   ########################
dir = "C:/Users/lelise/NewRiver_Local/lisflood/03_output/NR_30m/subgrid_bathy_v2_q/"
stations_file = "C:/Users/lelise/Desktop/lisflood_extract_output.csv"
ext = c('Qy','Qx','Qcy','Qcx')
tstart = "2018/09/07 00:00"
tz = "UTC"
timestep = '1 hour'
saveoutput=0
cell_size = 30

#################    Execute   #####################
# Save output as Rdata
if (saveoutput == 1){
  # Create a master list for all model output to be written to
  output = list()
  for (j in 1:length(ext)){
    output[[j]] = list()
  }
  
  # Write model output to master list
  for (j in 1:length(ext)){
    files = get_filenames(dir, ext[j])
    print(ext[j])
    output[[j]]=get_flux(files,
                         stations.df = read.csv(stations_file),
                         epsg = "32119" )
  }
  save(output, file = paste0(dir,'florence_flux.RData'))
}else {
  load(file = paste0(dir,'florence_flux.RData'))
}

#################    Plotting   #####################
for (j in 1:length(output[[1]])){
  for (i in 1:length(ext)){
    data = output[[i]][j]*cell_size#*35.3147
    if (i==1){
      data_ts1 = xts(x=data, 
                     order.by = seq.POSIXt(from = as.POSIXct(tstart,tz),
                                           by = timestep,
                                           length.out = nrow(output[[1]])))}
    else {
      data_ts = xts(x=data, 
                    order.by = seq.POSIXt(from = as.POSIXct(tstart,tz),
                                          by = timestep,
                                          length.out = nrow(output[[1]])))
      data_ts1 = merge(data_ts1, data_ts)
    }
  }
  colnames(data_ts1) = ext
  
  station_name = colnames(data)
  p = plot(data_ts1,lty=1, lwd=3,cex=0.7,
       yaxis.right=FALSE, main=station_name, legend.loc='topleft',
       ylabel='Flux (cfs)')
  png(filename = paste0(dir,station_name,'.png'))
  print(p)
  dev.off()
  
  write.csv(as.data.frame(data_ts1),
            paste0(dir,station_name,'.csv'),
            row.names = TRUE)
}

autoplot(data_ts1, geom='line',
         ylab = 'Flux (cfs)',
         legend = TRUE,
         main = station_name)

ras <- raster("C:/Users/lelise/NewRiver_Local/lisflood/03_output/NR_30m/subgrid_bathy_v2_q/florence.max")
crs(ras) <- CRS('+init=EPSG:32119')
ras[ras < 0.25] = NA
writeRaster(ras,
            "C:/Users/lelise/NewRiver_Local/lisflood/03_output/NR_30m/subgrid_bathy_v2_q/florence_max.tif",
            format ='GTiff',
            overwrite=TRUE)