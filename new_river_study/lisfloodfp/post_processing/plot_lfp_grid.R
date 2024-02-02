library(ggplot2)
library(raster)

file = 'C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output\\florence-m5-surge.max'


ras <- raster(file)
crs(ras) <- CRS('+init=EPSG:32119')
r2 = trim(ras)
plot(r2)

writeRaster(ras,
            "C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\florence_m5_max_surge.tif",
            format ='GTiff',
            overwrite=TRUE)



#PLOT OBSERVED V MODELED HWM
library(gdal)
library(raster)
library(RColorBrewer)

#import raster
mxe <- raster('C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output\\florence-m1.mxe')
max <- raster('C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output\\florence-m1.max')

#assign color to a object for repeat use/ ease of changing
myCol = brewer.pal(n = 5, name = "PuBu")

#plot raster
plot(max, 
     breaks = c(0.25, 0.5, 1, 2, 10), 
     col = myCol,
     main="Maximum Water Depth (m)\n New River Watershed, Hurricane Florence (2018)", 
     xlab = "UTM Westing Coordinate (m)", 
     ylab = "UTM Northing Coordinate (m)")

#plot vector data
#hwm <- read.csv("01_data/USGS/FilteredHWMs.csv")
#hwm <- readOGR