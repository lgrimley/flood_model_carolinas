library(VineCopula)
library(copBasic)
library(fitdistrplus)
library(fBasics)
library(ggplot2)
library(kdecopula)
library(patchwork)
library(metR)

setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\kde_copula\\MeanTotPrecip_vs_AvgMaxWS')
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

climates = list('ncep','canesm')
lambda_atlantic = 3.38
n_storms = list(5018, 6200)
# yvar = 'AvgmaxRR'
# xvar = 'stormtide'
# ylabel = 'Area-Avg Max Rain Rate\n(mm/hr)'
# xlabel = 'Storm Tide (m)'
yvar = 'MeanTotPrecipMM'
xvar = 'meanMaxWS'
ylabel = 'Area-Avg Total\nPrecipitation (mm)'
xlabel = 'Area-Avg Max Wind Speeds (m/s)'
basins <- c('CapeFear','Pamlico','Neuse','OnslowBay','LowerPeeDee','Domain')

# Climate loop
kk=1
results_list <- list() 
plot_list <- list()
climate <- climates[[kk]]
storms_atlantic <- n_storms[[kk]]

print(climate)
print(lambda_atlantic)
print(storms_atlantic)
print(yvar)
print(xvar)

ii=1
b <- basins[[ii]]

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
    
#### Fit KDE Copula ######
# Transform the data to pseudo observations
u <- pobs(x)
v <- pobs(y)
uv_mat = cbind(u,v)

# Tail Dependence Estimation
# Calculate coefficients that are probabilities ranging from 0 to 1
# lambda = 0 is no dependence, lambda = 1 is strong dependence
alpha <- 0.10 
print(paste0('Tail dependence for alpha = ', alpha))
# Upper Tail
lambda_U <- mean(v[u > 1 - alpha] > 1 - alpha)
print(paste0('Upper tail lambda = ', lambda_U))
# Lower Tail
lambda_L <- mean(v[u < alpha] < alpha)
print(paste0('Lower tail lambda = ', lambda_L))

#### Compute the Joint CDF using the kdecopula
emp_cop <- kdecop(udata=uv_mat)
print(emp_cop)
plot(emp_cop, display = "3D")
contour(emp_cop, margins = "unif")

# calculate dependence measures
dep_measures(emp_cop)

# Check the copula works and makes sense and save output
summary(emp_cop)
plot(emp_cop)
# Should be close to 0
pkdecop(matrix(c(0, 0), ncol=2), emp_cop)
# Should be close to 1
pkdecop(matrix(c(1, 1), ncol=2), emp_cop)
# Median point should be ~0.25 if copula is close to independent
pkdecop(matrix(c(0.5, 0.5), ncol=2), emp_cop)

# Extract outputs
fit_info <- list(
  climate = climate,
  xvar = xvar,
  yvar = yvar,
  basin_name = b,
  lambda_basin = lambda_basin,
  nstorms = storms_basin,
  tail_alpha = alpha,
  lambda_U = lambda_U,
  lambda_L = lambda_L,
  loglik = emp_cop$info$loglik,
  effp = emp_cop$info$effp,
  AIC = emp_cop$info$AIC,
  cAIC = emp_cop$info$cAIC,
  BIC = emp_cop$info$BIC
)
#results_list[[ii]] <- as.data.frame(fit_info, stringsAsFactors = FALSE)

#### Apply KDE copula
# Grid of u,v values
grid <- expand.grid(u = seq(0, 1, length.out = 100),
                    v = seq(0, 1, length.out = 100))

# KDE copula CDF values
grid$cop <- pkdecop(as.matrix(grid), emp_cop)

ggplot(grid, aes(x = u, y = v, z = cop)) +
  geom_contour(bins = 10) +
  labs(title = "Contour of Fitted KDE Copula CDF",
       x = "u", y = "v") +
  theme_minimal()

# Generate a joint probability grid that covers the variable space
Xvec <- seq(min(x), max(x), length.out = 300)
Yvec <- seq(min(y), max(y), length.out = 300)
XY_mat <- expand.grid(Xvec, Yvec)
summary(XY_mat)

# Option 1: Get smoothed spline estimate of the marginal CDF
# Get the CDF value of the spline at a point (fBasics)
x_marg <- ssdFit(x)
y_marg <- ssdFit(y)
u_grid <- pssd(XY_mat$Var1, x_marg)
v_grid <- pssd(XY_mat$Var2, y_marg)
# Make sure the transformed values are between 0 and 1
summary(u_grid)
summary(v_grid)

# Option 2: fitting marginal dist using parametric
#fit_x <- fitdist(x, "gamma")
#fit_y <- fitdist(y, "gamma")
#u_grid <- pgamma(XY_mat$Var1, shape=fit_x$estimate["shape"], rate=fit_x$estimate["rate"])
#v_grid <- pgamma(XY_mat$Var2, shape=fit_y$estimate["shape"], rate=fit_y$estimate["rate"])

# Matrix with 2 columns with values between 0 and 1
uv_pseudo_mat<- cbind(u_grid, v_grid)

# Compute the joint CDF
# second argument is a fitted kernal copula object
cdf_grid  <- pkdecop(uv_pseudo_mat, emp_cop)

str(uv_pseudo_mat)
summary(cdf_grid)

# This should return true otherwise grid has values outside 0 or 1
all(cdf_grid >= 0 & cdf_grid <= 1)

# Visualize
XY_mat$cdf <- cdf_grid

ggplot(XY_mat, aes(x = Var1, y = Var2, fill = cdf)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Joint CDF from KDE Copula", x = xlabel, y = ylabel) +
  theme_minimal()

# Get an estimate of the joint exceedance probability
# 1 - (P(X <= x)) - P(Y <= y) + P(X <= x, Y <=y)
P_AND <- 1 - u_grid - v_grid + cdf_grid
# Values should be between 0 and 1
P_AND <- pmax(P_AND, 0)
summary(P_AND)

# Add return period grid and check values
rp_grid <- 1 / (lambda_basin * P_AND)
# remove Inf or NaN values
rp_grid[is.infinite(rp_grid)] <- NA
summary(rp_grid)
XY_mat$return_period <- rp_grid
XY_mat$return_period <- round(XY_mat$return_period, 1)



#### Fit Parametric Copula ####
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
                        familyset = c(2), 
                        selectioncrit = "BIC",
                        #indeptest=TRUE,
                        #level=0.05,
                        rotations=TRUE,
                        method="mle"
)
print(summary(fit_tail))

fit <- fit_tail

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
fit_info_parametric <- list(
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
fit_info_parametricv <- lapply(fit_info_parametric, function(x) if (length(x) == 0) NA else x)
#results_list[[ii]] <- as.data.frame(fit_info, stringsAsFactors = FALSE)


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
} else if (fit$familyname == 'Gumbel'){
  copula <- GHcop #copBasic
} else if (fit$familyname == 'Student'){
  copula <- tEVcop #copBasic
}

joint_probs_or <- joint.curvesCOP(cop=copula, para = fit$par, 
                                  type = 'and', probs = target_prob)
joint_probs_and <- joint.curvesCOP(cop=copula, para = fit$par, 
                                   type = 'or', probs = target_prob)

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



#### Expected ####
expected_pts = storms_basin * (1/(100 * lambda_basin))
print(paste0('Expected poinst above 100-yr contour: ', expected_pts))

#### Plotting #####
df <- clean_data[order(clean_data$Compound_Area_sqkm_RP), ]
df$RP <- cut(df$Compound_Area_sqkm_RP,
             breaks = c(0, 10, 50, 100, 250, Inf),
             labels = c("<10yr", "10-50yr", "50-100yr", "100-250yr", '>250yr'),
             include.lowest = TRUE)
df$RP = factor(df$RP)

# Subset points where RP is around 100 (90–110)
df_outline <- df[df$Compound_Area_sqkm_RP > 90 & df$Compound_Area_sqkm_RP < 110, ]

# Parametric line
line_data_100 <- data.frame(x = x_100yr_and, y = y_100yr_and)


# Create custom blue shades (light to dark)
blue_shades <- c("gainsboro", "#bdd7e7", "#6baed6","#2171b5", "#084594", "#021e40") #"#4292c6"

p <- ggplot() +
  geom_point(data=df, aes(x = .data[[xvar]], y = .data[[yvar]], color = RP), 
             size=2.5, shape=16) +
  scale_color_manual(values = blue_shades, name='Return Period') +
  
  geom_point(data=df_outline,
             aes(x = .data[[xvar]], y = .data[[yvar]]), 
             shape = 21, fill = NA, color = "black", 
             stroke = 1.2, size = 3.75, show.legend = FALSE) +
  
  geom_contour(data=XY_mat, aes(x = Var1, y = Var2, z = return_period),
               breaks = c(100), color='black', size=1, linetype="longdash", alpha=0.8) +
  
  geom_line(data = line_data_100, aes(x = x, y = y), 
            color = 'green', size = 1, linetype = "longdash", alpha = 0.8) +
  
  
  labs(title = paste0(b,' (n=',storms_basin,')'), x = xlabel, y = ylabel) +
  theme_minimal()+ theme(text = element_text(size = 10))

figname = paste0(b,'_',climate,'_KDECopula.png')
#ggsave(figname, plot = p, width = 5, height = 3.5, dpi = 300, bg='white')









