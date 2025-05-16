library(VineCopula)
library(fitdistrplus)
library(fBasics)
library(ggplot2)
library(kdecopula)
library(metR)

# load and process data 
setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\')
basins = c('CapeFear','LowerPeeDee', 'Neuse', 'Pamlico', 'OnslowBay', 'Domain')

# User Input
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'
climate = 'canesm'
# Atlantic basin arrival rate
ab_lambda = 3.38
ab_storms = 6200 
# total number of TCs in study area
total = 1543 
# TC arrival rate
lambda = ab_lambda * (total/ab_storms)
# return periods and probabilities
RPs = c(10, 25, 50, 100, 250, 500, 1000)
cum_probs <- 1 + log(1-1/RPs)/lambda

for (b in basins){
  filename = paste0(datadir, '\\', b, '_data_rp_', climate,'.csv')
  storm_metrics = read.csv(filename)
  storm_metrics <- as.data.frame(storm_metrics)
  
  # look at correlation between the data
  clean_data1 <- storm_metrics[!is.na(storm_metrics$stormtide),]  # remove the rows with NA
  x = clean_data1$stormtide
  y = clean_data1$maxRR
  
  storm_metrics[is.na(storm_metrics)] <- 0 # replace the NA with zero
  clean_data <- storm_metrics
  
  ###########################################################################
  
  # fit marginal distributions using smoothed spline density estimate 
  # using fBasics package; Fits parameters for a spline smoothed distribution
  x_marg <- ssdFit(x)
  y_marg <- ssdFit(y)
  
  # transform data to pseudo observations (VineCopula or copula packages)
  u <- pobs(x)
  v <- pobs(y)
  
  # Get the empirical copula (kdecopula package)
  mat <- cbind(u, v)
  emp_cop <- kdecop(mat)
    
  # pssd gives the distribution function (fBasics package)
  xvec <- pssd(seq(0, max(x), by=0.2), x_marg)
  x_RP <- 1/(1-exp(-lambda*(1-xvec)))
  yvec <- pssd(seq(min(y), max(y), by=5), y_marg)
  y_RP = 1/(1-exp(-lambda*(1-yvec)))
  xy_mat <- expand.grid(xvec, yvec) # put the x and y vector into a grid
  xy_mat <- as.matrix(xy_mat) # enforce it as a matrix
  xy_mat <- apply(xy_mat, 2, as.numeric) # set as numeric values
  
  ###########################################################################
  
  # Get the CDF of the KDE copula
  cdf <- pkdecop(xy_mat, emp_cop)
  
  # Get the joint probability
  P_AND <- 1-xy_mat[,1]-xy_mat[,2] + cdf
  P_OR <- 1-cdf
  
  # Generate vectors of data to build the contours with
  Xvec <- seq(0, max(x), by=0.2)
  Yvec <- seq(min(y), max(y), by=5)
  XY_mat <- expand.grid(Xvec, Yvec)
  
  # build contours for the AND + OR scenario
  C_and <- ggplot_build(ggplot() + geom_contour(aes(x=XY_mat[,1], y=XY_mat[,2], z=P_AND)))
  #C_or <- ggplot_build(ggplot() + geom_contour(aes(x=XY_mat[,1], y=XY_mat[,2], z=P_OR)))
  contour_data <- data.frame(
    x = XY_mat[, 1],
    y = XY_mat[, 2],
    z = P_AND  # or use P_OR
  )
  summary(contour_data$z)
  
  ###########################################################################
  df <- clean_data[order(clean_data$Compound_Area_sqkm_RP), ]
  df$RP <- cut(df$Compound_Area_sqkm_RP,
               breaks = c(0, 10, 25, 50, 100, 500, Inf),
               labels = c("<=10yr", "25yr", "50yr", "100yr", "500yr", '>500yr'),
               include.lowest = TRUE)
  df$RP = factor(df$RP)
  # Subset points where RP is around 100 (90–110)
  df_outline <- df[df$Compound_Area_sqkm_RP > 90 & df$Compound_Area_sqkm_RP < 110, ]
  # Create custom blue shades (light to dark)
  blue_shades <- c("#eff3ff", "#bdd7e7", "#6baed6", "#4292c6", "#2171b5", "#084594", "#021e40")
  joint_probs = 1-cum_probs[c(2, 4)] 
  p = ggplot() +
    geom_contour(data = contour_data, aes(x = x, y = y, z = z), 
                 breaks = joint_probs, color = "black", size = 1) +
    
    # Add manual text labels for 25-year and 100-year contours
    geom_text(aes(x = 0.25, y = 55, label = "25yr"), size = 5, color = "black") +
    geom_text(aes(x = 0.25, y = 80, label = "100yr"), size = 5, color = "black") +
    
    geom_point(data = df, aes(x = stormtide, y = maxRR, color = RP), size = 2.5) +
    geom_point(data = df_outline, aes(x = stormtide, y = maxRR),
               shape = 21, fill = NA, color = "black", stroke = 1.2, size = 4) +
    
    scale_color_manual(values = blue_shades) +
    labs(
      y = 'Rainfall Intensity (mm/hr)',
      x = 'Storm Tide Peak (m)',
      title = paste0(b, ' ', climate, ' (KDE)'),
      color = "Compound\nExtent\nReturn Period"
    ) +
    theme_minimal() +
    theme(axis.line = element_line(color = "black", size = 0.5))
  
  figname = paste0(b,'_',climate,'_KDECopula.png')
  ggsave(figname, plot = p, width = 6, height = 4, dpi = 300, bg='white')

} 

