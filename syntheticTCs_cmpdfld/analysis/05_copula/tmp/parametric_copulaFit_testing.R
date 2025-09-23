library(VineCopula)
library(copBasic)
library(fitdistrplus)
library(fBasics)
library(ggplot2)
library(VC2copula)
library(metR)
library(patchwork)
suppressWarnings(log(-1))



setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\parametric_copulas')
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

climates = list('ncep','canesm')
lambda_atlantic = 3.38
n_storms = list(5018, 6200)
#yvar = 'AvgmaxRR'
#xvar = 'stormtide'
#ylabel = 'Area-Avg Max Rain Rate\n(mm/hr)'
#xlabel = 'Storm Tide (m)'
yvar = 'MeanTotPrecipMM'
xvar = 'meanMaxWS'
ylabel = 'Area-Avg Total\nPrecipitation (mm)'
xlabel = 'Area-Avg Max Wind Speeds (m/s)'
basins <- c('Neuse')


#for (kk in seq_along(climates)){
kk = 1
results_list <- list() 
plot_list <- list()
climate <- climates[[kk]]
storms_atlantic <- n_storms[[kk]]

#sink(paste0(climate,"_copula_output.txt"))
print(climate)
print(lambda_atlantic)
print(storms_atlantic)
print(yvar)
print(xvar)

#for (ii in seq_along(basins)){
ii=1
b <- basins[[ii]]

print('###############################################')
print(b)
print('###############################################')

filename = paste0(datadir, '\\', b, '_data_rp_', climate,'.csv')
storm_metrics = read.csv(filename)
storm_metrics <- as.data.frame(storm_metrics)

#### Load Data #####
filename = paste0(datadir, '\\', b, '_data_rp_', climate,'.csv')
storm_metrics = read.csv(filename)
storm_metrics <- as.data.frame(storm_metrics)
# remove the rows with NA and zero (Avi said these should be null)
clean_data <- storm_metrics[
  !is.na(storm_metrics[[xvar]]) &
    storm_metrics[[xvar]] != 0,
]  
x = clean_data[[xvar]]
y = clean_data[[yvar]]
  
storms_basin = length(x)
print(paste0('Number of storms with data : ', storms_basin))

lambda_basin = lambda_atlantic * (storms_basin/storms_atlantic)
print(paste0('Arrival rate : ', lambda_basin))

# define return period  levels
target_rp <- c(10, 25, 50, 100, 250, 500, 1000)
# Cumulative probability
target_prob <- 1 + log(1 - 1 / target_rp) / lambda_basin
# Exceedance probability
target_exceed_prob <- 1 - target_prob

ggplot()+
  geom_point(data=clean_data, aes(x=.data[[xvar]], y=.data[[yvar]]), alpha=0.5) +
  labs(title=storms_basin, x=xlabel,y=ylabel) +
  theme_minimal()
  
# Calculate correlation stats between variables
spearrho = cor(x, y, method = 'spearman')
print('Spearmans rho (+1 is perfect increasing monotonic relationship, -1 is decreasing, 0 is no')
print(spearrho)

pearson = cor(x, y, method = 'pearson')
print('Pearson correlation coefficient (+1 is perfect positive linear relationship, -1 is negative, 0 is no')
print(pearson)

kendtau = cor(x, y, method = 'kendall')
print('Kendalls tau (+1 is perfect agreement between pairs, -1 is perfect disagreement, 0 is no association')
print(kendtau)

#### Fit Copula ######
# transform data to pseudo observations (VineCopula or copula)
u <- pobs(x)
v <- pobs(y)

# find the best Copula family using BIC criteria (VineCopula)
fitBIC <- BiCopSelect(u1=u, 
                   u2=v, 
                   familyset = NA, 
                   selectioncrit = "BIC",
                   indeptest=TRUE,
                   level=0.05,
                   method="mle"
                   )
print(summary(fitBIC))

fitAIC <- BiCopSelect(u1=u, 
                      u2=v, 
                      familyset = NA, 
                      selectioncrit = "AIC",
                      indeptest=TRUE,
                      level=0.05,
                      method="mle"
)
print(summary(fitAIC))

fit_tail <- BiCopSelect(u1=u, 
                      u2=v, 
                      familyset = c(3,4), 
                      selectioncrit = "BIC",
                      #indeptest=TRUE,
                      #level=0.05,
                      rotations=TRUE,
                      method="mle"
)
print(summary(fit_tail))

fit <- fitBIC
  
if (fit$familyname != 'Survival BB8'){
  # test goodness of fit of copula (VineCopula package)
  test_fit_white <-BiCopGofTest(u1=u, u2=v,
                                family=fit$family, 
                                par=fit$par, 
                                par2=fit$par2,
                                method = 'white',
                                B=300
                                )
  print('White’s information matrix equality. Returns asymptotic p-value and observed test statistic (Wn).')
  print('Wn measures how far your estimated information matricies deviate from equality.')
  print('Model may not fit well if p-value less than 0.05')
  print(test_fit_white)
}

test_fit_kendall <-BiCopGofTest(u1=u, u2=v, 
                                family=fit$family, 
                                par=fit$par, 
                                par2=fit$par2,
                                method = 'kendall', 
                                B=300 # number of bootstrap samples
                                )
print('Kendalls process. Returns Cramer-von Mises (CvM) and Kolmogorov-Smirnov (KS) test statistic.')
print('A small test statistic with a high p-value → fail to reject the null → model is a good fit.')
print('A large test statistic with a low p-value → reject the null → model is likely misspecified.')
print(test_fit_kendall)

  
# Extract outputs
fit_info <- list(
  climate = climate,
  xvar = xvar,
  yvar = yvar,
  basin_name = b,
  lambda_basin = lambda_basin,
  nstorms = storms_basin,
  family = fit$famiy,
  family_name = fit$familyname,
  tau = fit$tau,
  emptau = fit$emptau,
  pvalue_indeptest = fit$p.value.indeptest,
  lambda_U = fit$taildep$upper,
  lambda_L = fit$taildep$lower,
  loglik = fit$loglik,
  effp = fit$effp,
  AIC = fit$AIC,
  cAIC = fit$cAIC,
  BIC = fit$BIC,
  pvalue_CvM = test_fit_kendall$p.value.CvM,
  statistic_CvM = test_fit_kendall$statistic.CvM,
  pvalue_KS = test_fit_kendall$p.value.KS,
  statistic_KS = test_fit_kendall$statistic.KS,
  pvalue_White = test_fit_white$p.value[1], 
  statistic_White = test_fit_white$statistic[1]
)
fit_info <- lapply(fit_info, function(x) if (length(x) == 0) NA else x)
results_list[[ii]] <- as.data.frame(fit_info, stringsAsFactors = FALSE)
  
  
cop_fit <- BiCop(fit$family, fit$par, fit$par2)
print(summary(cop_fit))

#### Get copula contours ####
# compute joint return period curves using 'AND' or 'OR' criteria. NOTE: type='and' corresponds to 
# 'OR' scenario in terms of exceedance probability (i.e. failure probability)
# Once you fit the copula, select the right function from the copBasic package
if (fit$familyname == 'Survival Joe' || fit$familyname == 'Joe'){
  copula <- JOcopB5 #Vinecopula
} else if (fit$familyname == 'Independence'){
  copula <- P #copBasic
} else if (fit$familyname == 'Clayton' || fit$familyname == 'Survival Clayton'){
  copula <- CLcop #copBasic
} else if (fit$familyname == 'Frank'){
  copula <- FRcop
} else if (fit$familyname == 'Gaussian'){
  copula <- gEVcop #copBasic
}

joint_probs_or <- joint.curvesCOP(cop=copula, para = fit$par, 
                                  type = 'and', probs = target_prob)
joint_probs_and <- joint.curvesCOP(cop=copula, para = fit$par, 
                                   type = 'or', probs = target_prob)
  
# fit marginal distributions using smoothed spline density estimate 
# using fBasics package; Fits parameters for a spline smoothed distribution
x_marg <- ssdFit(x)
y_marg <- ssdFit(y)

# Find the closest probability
jointprobs <- as.numeric(names(joint_probs_and))

# Find the index of the closest value to the 100-yr
# Get the name of the closest match
target_100yr = target_prob[6]
closest_index <- which.min(abs(jointprobs - target_100yr))
closest_prob_name <- names(joint_probs_and)[closest_index]
pdf_100yr_and <- BiCopPDF(u1=joint_probs_and[[closest_prob_name]]$U, 
                          u2=joint_probs_and[[closest_prob_name]]$V, 
                          family=fit$family, par=fit$par)

# find values corresponding to quantiles from joint return period curves
# using fBasics package; Returns spline smoothed quantile function
x_100yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$U, param=x_marg)
y_100yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$V, param=y_marg)
  
# Find the index of the closest value to the 25-yr
# Get the name of the closest match
# target_25yr = target_prob[2]
# closest_index <- which.min(abs(jointprobs - target_25yr))
# closest_prob_name <- names(joint_probs_and)[closest_index]
# pdf_25yr_and <- BiCopPDF(u1=joint_probs_and[[closest_prob_name]]$U, 
#                          u2=joint_probs_and[[closest_prob_name]]$V, 
#                          family=fit$family, par=fit$par)
# 
# # find values corresponding to quantiles from joint return period curves
# # using fBasics package; Returns spline smoothed quantile function
# x_25yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$U, param=x_marg)
# y_25yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$V, param=y_marg)

  
########################################################
#### Plotting #####
df <- clean_data[order(clean_data$Compound_Area_sqkm_RP), ]

df$RP <- cut(df$Compound_Area_sqkm_RP,
                breaks = c(0, 10, 50, 100, 250, Inf),
                labels = c("<10yr", "10-50yr", "50-100yr", "100-250yr", '>250yr'),
                include.lowest = TRUE)
df$RP = factor(df$RP)

# Subset points where RP is around 100 (90–110)
df_outline <- df[df$Compound_Area_sqkm_RP > 90 & df$Compound_Area_sqkm_RP < 110, ]
line_data_100 <- data.frame(x = x_100yr_and, y = y_100yr_and)

# Create custom blue shades (light to dark)
blue_shades <- c("gainsboro", "#bdd7e7", "#6baed6","#2171b5", "#084594", "#021e40") #"#4292c6"

p <- ggplot() +
  geom_point(data=df, aes(x = .data[[xvar]], y = .data[[yvar]], color = RP), size=2.5, shape=16) +
  scale_color_manual(values = blue_shades, name='Return Period') +
  
  geom_point(data=df_outline,
             aes(x = .data[[xvar]], y = .data[[yvar]]), 
             shape = 21, fill = NA, color = "black", 
             stroke = 1.2, size = 3.75, show.legend = FALSE) +

  geom_line(data = line_data_100, aes(x = x, y = y), 
            color = 'black', size = 1, linetype = "longdash", alpha = 0.8) +
  #geom_line(aes(x=x_25yr_and, y=y_25yr_and), color='black', size=1, linetype="longdash", alpha=0.8) +
  
  # Add text label near the end of each line
  # geom_text(aes(x = tail(x_100yr_and, 1), y = tail(y_100yr_and, 1), label = "100yr"),
  #           hjust = -0.1, vjust = -0.1, size = 4) +
  #geom_text(aes(x = tail(x_25yr_and, 1), y = tail(y_25yr_and, 1), label = "25yr"),
            #hjust = -0.1, vjust = -0.1, size = 4) +
  
  labs(title = paste0(b,' (n=',storms_basin,')'), x = xlabel, y = ylabel) +
  theme_minimal()+ theme(text = element_text(size = 10))


figname = paste0(b,'_',climate,'_paramCopula.png')
ggsave(figname, plot = p, width = 5, height = 3.5, dpi = 300, bg='white')
plot_list[[ii]] <- p
p

#sink()
#combined_plot <- wrap_plots(plot_list, ncol = 2)+
#  plot_layout(guides = "collect") & 
#  theme(legend.position = "bottom")
#ggsave(paste0('all_basins_copula_grid_',climate,".png"), combined_plot, width = 6.5, height = 8, dpi = 300, bg = "white")

#results_df <- do.call(rbind, results_list)
#results_df[] <- lapply(results_df, function(x) {
#  if (is.numeric(x)) round(x, 3) else x
#})
#write.csv(results_df, file = paste0('paramCopula_info_',climate,".csv"), row.names = FALSE)
#}

##################################
# climate = 'canesm'
# # Atlantic basin arrival rate
# ab_lambda = 3.38
# ab_storms = 6200
# 
# 
# sink(paste0(climate,"_copula_output.txt"))
# 
# for (b in basins){
#   print('###############################################')
#   print(b)
#   print('###############################################')
#   filename = paste0(datadir, '\\', b, '_data_rp_', climate,'.csv')
#   storm_metrics = read.csv(filename)
#   storm_metrics <- as.data.frame(storm_metrics)
#   
#   # look at correlation between the data
#   clean_data <- storm_metrics[!is.na(storm_metrics$stormtide), ]  # remove the rows with NA
#   clean_data <- clean_data[clean_data$stormtide != 0, ]  # remove the rows equal to zero
#   x = clean_data[[xvar]]
#   y = clean_data[[yvar]]
#   
#   # total number of TCs in study area
#   n_storms = length(x)
#   print('Number of storms with data')
#   print(n_storms)
#   
#   # TC arrival rate
#   lambda = ab_lambda * (n_storms/ab_storms)
#   print('Arrival rate')
#   print(lambda)
#   
#   # return periods and probabilities
#   RPs = c(10, 25, 50, 100, 250, 500, 1000)
#   cum_probs <- 1 + log(1-1/RPs)/lambda
#   
#   plot(x, y)
#   #abline(lm(x ~ y), col='red', lwd=1)
#   
#   # Calculate correlation stats
#   spearrho = cor(x, y, method = 'spearman')
#   print('Spearmans rho (+1 is perfect increasing monotonic relationship, -1 is decreasing, 0 is no')
#   print(spearrho)
#   
#   pearson = cor(x, y, method = 'pearson')
#   print('Pearson correlation coefficient (+1 is perfect positive linear relationship, -1 is negative, 0 is no')
#   print(pearson)
#   
#   kendtau = cor(x, y, method = 'kendall')
#   print('Kendalls tau (+1 is perfect agreement between pairs, -1 is perfect disagreement, 0 is no association')
#   print(kendtau)
#   
#   # transform data to pseudo observations (VineCopula or copula package)
#   u <- pobs(x)
#   v <- pobs(y)
#   
#   # find the best Copula family using BIC criteria (VineCopula package)
#   fit1 <- BiCopSelect(u1=u, 
#                       u2=v, 
#                       familyset = NA, 
#                       selectioncrit = "BIC",
#                       indeptest=TRUE,
#                       method="mle"
#   )
#   print(summary(fit1))
#   
#   # find the best Copula family using BIC criteria (VineCopula package)
#   fit <- BiCopSelect(u1=u, 
#                      u2=v, 
#                      familyset = c(0,1,2,3,4,5,6,13,16,23,24,26,33,34,36), 
#                      selectioncrit = "BIC",
#                      indeptest=TRUE,
#                      method="mle"
#   )
#   print(summary(fit))
#   
#   # Calculate tail dependence coefficients
#   tail_dep_coeff <- BiCopPart2TailDep(family=fit$family, par=fit$par, par2=fit$par2)
#   print(tail_dep_coeff)
#   
#   # test goodness of fit of copula (VineCopula package)
#   if (fit$familyname != 'Survival BB8'){
#     
#     test_fit_white <-BiCopGofTest(u1=u, u2=v,
#                                   family=fit$family, 
#                                   par=fit$par, 
#                                   par2=fit$par2,
#                                   method = 'white',
#                                   B=300
#     )
#     print('White’s information matrix equality. Returns asymptotic p-value and observed test statistic (Wn).')
#     print('Wn measures how far your estimated information matricies deviate from equality.')
#     print('Model may not fit well if p-value less than 0.05')
#     print(test_fit_white)
#   }
#   
#   test_fit_kendall <-BiCopGofTest(u1=u, u2=v, 
#                                   family=fit$family, 
#                                   par=fit$par, 
#                                   par2=fit$par2,
#                                   method = 'kendall', 
#                                   B=300 # number of bootstrap samples
#   )
#   print('Kendalls process. Returns Cramer-von Mises (CvM) and Kolmogorov-Smirnov (KS) test statistic.')
#   print('A small test statistic with a high p-value → fail to reject the null → model is a good fit.')
#   print('A large test statistic with a low p-value → reject the null → model is likely misspecified.')
#   print(test_fit_kendall)
#   
#   cop_fit <- BiCop(fit$family, fit$par, fit$par2)
#   print(summary(cop_fit))
#   
#   # compute joint return period curves using 'AND' or 'OR' criteria. NOTE: type='and' corresponds to 
#   # 'OR' scenario in terms of exceedance probability (i.e. failure probability)
#   # Once you fit the copula, select the right function from the copBasic package
#   if (fit$familyname == 'Survival Joe' || fit$familyname == 'Joe'){
#     copula <- JOcopB5 #Vinecopula
#   } else if (fit$familyname == 'Independence'){
#     copula <- P #copBasic
#   } else if (fit$familyname == 'Clayton' || fit$familyname == 'Survival Clayton' || fit$familyname == 'Survival Gumbel'){
#     copula <- CLcop #copBasic
#   } else if (fit$familyname == 'Frank'){
#     copula <- FRcop
#   } else if (fit$familyname == 'Gaussian'){
#     copula <- gEVcop #copBasic
#   }
#   
#   joint_probs_or <- joint.curvesCOP(cop=copula, para = fit$par, 
#                                     type = 'and', probs = cum_probs)
#   joint_probs_and <- joint.curvesCOP(cop=copula, para = fit$par, 
#                                      type = 'or', probs = cum_probs)
#   
#   # fit marginal distributions using smoothed spline density estimate 
#   # using fBasics package; Fits parameters for a spline smoothed distribution
#   x_marg <- ssdFit(x)
#   y_marg <- ssdFit(y)
#   
#   # Find the closest probability
#   jointprobs <- as.numeric(names(joint_probs_and))
#   
#   # Find the index of the closest value to the 100-yr
#   # Get the name of the closest match
#   target_prob = cum_probs[6]
#   closest_index <- which.min(abs(jointprobs - target_prob))
#   closest_prob_name <- names(joint_probs_and)[closest_index]
#   pdf_100yr_and <- BiCopPDF(u1=joint_probs_and[[closest_prob_name]]$U, 
#                             u2=joint_probs_and[[closest_prob_name]]$V, 
#                             family=fit$family, 
#                             par=fit$par)
#   
#   # find values corresponding to quantiles from joint return period curves
#   # using fBasics package; Returns spline smoothed quantile function
#   x_100yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$U, 
#                       param=x_marg)
#   y_100yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$V, 
#                       param=y_marg)
#   
#   # Find the index of the closest value to the 25-yr
#   # Get the name of the closest match
#   target_prob = cum_probs[2]
#   closest_index <- which.min(abs(jointprobs - target_prob))
#   closest_prob_name <- names(joint_probs_and)[closest_index]
#   pdf_25yr_and <- BiCopPDF(u1=joint_probs_and[[closest_prob_name]]$U, 
#                            u2=joint_probs_and[[closest_prob_name]]$V, 
#                            family=fit$family, 
#                            par=fit$par)
#   
#   # find values corresponding to quantiles from joint return period curves
#   # using fBasics package; Returns spline smoothed quantile function
#   x_25yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$U, 
#                      param=x_marg)
#   y_25yr_and <- qssd(p=joint_probs_and[[closest_prob_name]]$V, 
#                      param=y_marg)
#   
#   
#   ########################################################
#   df <- clean_data[order(clean_data$Compound_Area_sqkm_RP), ]
#   
#   df$RP <- cut(df$Compound_Area_sqkm_RP,
#                breaks = c(0, 10, 50, 100, 250, Inf),
#                labels = c("<10yr", "10-50yr", "50-100yr", "100-250yr", '>250yr'),
#                include.lowest = TRUE)
#   df$RP = factor(df$RP)
#   
#   # Subset points where RP is around 100 (90–110)
#   df_outline <- df[df$Compound_Area_sqkm_RP > 90 & df$Compound_Area_sqkm_RP < 110, ]
#   
#   # Create custom blue shades (light to dark)
#   blue_shades <- c("gainsboro", "#bdd7e7", "#6baed6","#2171b5", "#084594", "#021e40") #"#4292c6"
#   
#   p <- ggplot() +
#     geom_point(data=df, aes(x = .data[[xvar]], y = .data[[yvar]], color = RP), size=2.5, shape=16) +
#     scale_color_manual(values = blue_shades) +
#     geom_point(data=df_outline,
#                aes(x = .data[[xvar]], y = .data[[yvar]]), 
#                shape = 21, fill = NA, color = "black", stroke = 1.2, size = 3.75) +
#     
#     geom_line(aes(x=x_100yr_and, y=y_100yr_and), color='black', size=1, linetype="longdash", alpha=0.8) +
#     geom_line(aes(x=x_25yr_and, y=y_25yr_and), color='black', size=1, linetype="longdash", alpha=0.8) +
#     
#     # Add text label near the end of each line
#     geom_text(aes(x = tail(x_100yr_and, 1), y = tail(y_100yr_and, 1), label = "100yr"),
#               hjust = -0.1, vjust = -0.1, size = 4) +
#     geom_text(aes(x = tail(x_25yr_and, 1), y = tail(y_25yr_and, 1), label = "25yr"),
#               hjust = -0.1, vjust = -0.1, size = 4) +
#     
#     labs(y=ylabel,
#          x=xlabel,
#          title=paste0('Projected Period: ', b, ' (', fit$familyname, ')'),
#          color = "Compound\nExtent\nReturn Period",
#     ) + 
#     theme_minimal() +
#     theme(
#       axis.line = element_line(color = "black", size = 0.5)  # Adding black axis lines
#     )
#   figname = paste0(b,'_',climate,'_paramCopula.png')
#   ggsave(figname, plot = p, width = 6, height = 4, dpi = 300, bg='white')
# }
# 
# sink()

  
