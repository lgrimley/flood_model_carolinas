library(raster)
library(RColorBrewer)
library(sf)
library(mapview)
library(tidyr)
library(ggplot2)
library(leaflet)
library(ggrepel)

#import raster
mxe <- raster("C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output/florence-m5.mxe")

crs(mxe) <- CRS('+init=EPSG:32119')

#get hwm data and plot
hwm.df <- read.csv("C:\\Users\\lelise\\OneDrive - University of North Carolina at Chapel Hill\\NewRiver\\scripts/FilteredHWMs.csv")

#Convert data frame to sf object
hwm.sf.point <- st_as_sf(x = hwm.df, 
                        coords = c("longitude", "latitude"),
                        crs = "+init=EPSG:4269")

#extract raster values to points
hwm.sf.point$rasValue = raster::extract(mxe, hwm.sf.point)

#drop NA
hwm.sf.point <- hwm.sf.point[!is.na(hwm.sf.point$rasValue),]


#plot observed v modeled data
ggplot(hwm.sf.point, aes(x = elev_ft*0.3048, y=rasValue, label=siteDescription)) + 
  geom_point(color='black', size=2) +
  geom_abline(linetype="dashed", lwd=1, alpha = 0.25) +
  geom_text_repel(box.padding=0.25, max.overlaps = Inf, size=2.5,
                  nudge_x=0.75, nudge_y=0.15)+
  labs(x="Observed Water Level (m)", y= "Modeled Water Level (m)") +
  ylim(1.5,10.5)+xlim(1.5,10.5)+
  theme_bw()+
  theme(text=element_text(size=7))
ggsave(filename="florence-m5-hwm.png",
       path='C:\\Users\\lelise\\UNC_Research\\NewRiver\\lisflood\\florence\\output/',
       plot(last_plot()),
       device='png',
       width = 4,
       height= 2,
       units='in',
       dpi = 600)
  

#add a column for the difference between modeled and observed
#positive value modeled>observed (overpredict)
hwm.sf.point$diff <- hwm.sf.point$rasValue-hwm.sf.point$elev_ft*0.3048
mapview(hwm.sf.point['diff'])
