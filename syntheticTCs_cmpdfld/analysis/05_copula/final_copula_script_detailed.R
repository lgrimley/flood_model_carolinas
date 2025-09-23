library(VineCopula)
library(copBasic)
#library(kdecopula)
library(fBasics)
library(ggplot2)
library(patchwork)
library(zoo)
library(fitdistrplus)


# Global settings
setwd(
  'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\manuscript_fig'
)

datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

basins <- c('Neuse', 'OnslowBay','Pamlico','LowerPeeDee', 'Domain', 'CapeFear')
climate = 'ncep'
nstorms = 5018
yvar = 'MeanTotPrecipMM'
xvar = 'meanMaxWS'
zvar = "Total_Area_sqkm_RP"
lambda_atl = 3.38

for (basin in basins) {
  filename = paste(basin,climate, yvar, xvar, zvar, sep="_")
  sink(paste0(filename,"_console_output.txt"), split = TRUE)

# Read in the data and clean it
filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
df <- read.csv(filepath)[, c(xvar, yvar, zvar)]
df <- na.omit(df)
head(df)

# Calculate the lambda and target prob
lambda_basin <- lambda_atl * (nrow(df) / nstorms)
target_rp <- 100 # Return period  levels
target_prob <- 1 + log(1 - 1 / target_rp) / lambda_basin
target_exceed_prob <- 1 - target_prob


#### Transform the data to pseudo observations ####
u <- pobs(df[[xvar]])
v <- pobs(df[[yvar]])
uv <- cbind(u, v)

# #### Fit the KDE copula to the pseudo obs ####
# emp_cop <- kdecop(udata = uv)
# print('Summary of KDE copula fit')
# summary(emp_cop)
#
# # 3D Plot
# # plot(emp_cop, display = "3D")
# # Contour Plot
# # contour(emp_cop, margins = "unif")
# cat("KDE copula val should be close to zero: ", pkdecop(matrix(c(0, 0), ncol=2), emp_cop), "\n")
# cat("KDE copula val should be close to one: ", pkdecop(matrix(c(1, 1), ncol=2), emp_cop), "\n")
# cat("Median point should be ~0.25 if copula is close to independent: ", pkdecop(matrix(c(0.5, 0.5), ncol=2), emp_cop), "\n")
#
# # save kde fit outputs
# kdefit_info <- list(
#     method = emp_cop$method,
#     alpha = emp_cop$bw$alpha,
#     kappa = emp_cop$bw$kappa,
#     loglik = emp_cop$info$loglik,
#     effp = emp_cop$info$effp,
#     AIC = emp_cop$info$AIC,
#     cAIC = emp_cop$info$cAIC,
#     BIC = emp_cop$info$BIC
#     )
#
# # calculate dependence measures
# dep_measures <- dep_measures(emp_cop)
# print('KDE dependence measures:')
# print(as.list(dep_measures))
#
# #### Fit marginal distributions ####
# # Smoothed kernel density
# marg_x <- ssdFit(df[[xvar]])
# marg_y <- ssdFit(df[[yvar]])
#
# #### Get kdecopula CDF ####
# Setup vector limits for grid to calculate probabilities
n_grid = 200
xvar_min = floor(min(df[[xvar]]))
xvar_max = ceiling(max(df[[xvar]]))
yvar_min = floor(min(df[[yvar]]))
yvar_max = ceiling(max(df[[yvar]]))
xvar_vec = seq(xvar_min, xvar_max, length.out = n_grid)
yvar_vec = seq(yvar_min, yvar_max, length.out = n_grid)

# Grid in the original variable units
grid_uv <- expand.grid(X = xvar_vec, Y = yvar_vec)

# Get the CDF values using smoothed spline estimate of the marginal CDF
# u_grid <- pssd(grid_uv$X, marg_x)
# v_grid <- pssd(grid_uv$Y, marg_y)

# # clip transformed values into (0, 1)
# epsilon <- 1e-6
# u_pseudo <- pmin(pmax(u_grid, epsilon), 1 - epsilon)
# v_pseudo <- pmin(pmax(v_grid, epsilon), 1 - epsilon)
#
# # Grid transformed into the marginal CDF values (pseudo-obs)
# # so copula operates on uniform data
# uv_grid_pseudo <- cbind(u_pseudo, v_pseudo)
#
# # Get KDE copula CDF values on uniform grid
# kdecop_cdf <- pkdecop(uv_grid_pseudo, emp_cop)
# print("KDE Copula CDF:")
# summary(kdecop_cdf)
# cat("This should return true otherwise grid has values outside 0 or 1: ", all(kdecop_cdf >= 0 & kdecop_cdf <= 1), "\n")

#### AND Probabilities ####
# # Get an estimate of the joint exceedance probability
# # 1 - (P(X <= x)) - P(Y <= y) + P(X <= x, Y <=y)
# P_AND <- 1 - u_grid - v_grid + kdecop_cdf
# # Probabilities should be between 0 and 1
# P_AND <- pmax(P_AND, 0)
# grid_uv$exceed_prob = P_AND
#
# # Calculcate return periods from exc prob and add to grid
# RP_AND <- round(1 / (lambda_basin * P_AND), 1)
# RP_AND[is.infinite(RP_AND)] <- NA
# grid_uv$return_period <- RP_AND


#### Parametric Copula ####
cat('\n', 'PARAMETRIC FIT', '\n')

# Function to fit all and return the best fit by AIC
# Candidate distributions and their corresponding CDF and quantile function names
candidate_dists <- list(
  gamma = list(cdf = pgamma, quantile = qgamma),
  lnorm = list(cdf = plnorm, quantile = qlnorm),
  norm  = list(cdf = pnorm,  quantile = qnorm),
  weibull = list(cdf = pweibull, quantile = qweibull),
  exp = list(cdf = pexp, quantile = qexp),
  logis = list(cdf = plogis, quantile = qlogis)
)
get_best_fit <- function(data) {
  fits <- lapply(names(candidate_dists), function(distname) {
    fit <- tryCatch(
      fitdist(data, distname),
      error = function(e) NULL
    )
    return(fit)
  })

  # Filter out NULLs
  fits <- Filter(Negate(is.null), fits)

  # Pick the best based on AIC
  best_fit <- fits[[which.min(sapply(fits, function(f) f$aic))]]
  best_name <- best_fit$distname
  list(
    name = best_name,
    fit = best_fit,
    cdf = candidate_dists[[best_name]]$cdf,
    quantile = candidate_dists[[best_name]]$quantile
  )
}
marg_x_p <- get_best_fit(df[[xvar]])
marg_y_p <- get_best_fit(df[[yvar]])
cat("Best-fit distribution for", xvar, ":", marg_x_p$name, "\n")
cat("Best-fit distribution for", yvar, ":", marg_y_p$name, "\n")

print(marg_x_p)
print(marg_y_p)

fit <- BiCopSelect(
  u1 = u,
  u2 = v,
  familyset = c(0,1,2,3,4,5,6),
  selectioncrit = "AIC",
  indeptest = TRUE,
  level = 0.05,
  method = "mle"
)
print('BiCopSelect copula fit:')
summary(fit)

# if (fit$familyname != 'Survival BB8') {
#   # test goodness of fit of copula (VineCopula package)
#   test_fit_white <- BiCopGofTest(
#     u1 = u,
#     u2 = v,
#     family = fit$family,
#     par = fit$par,
#     par2 = fit$par2,
#     method = 'white',
#     B = 100
#   )
#   print('White’s information matrix equality. Returns asymptotic p-value and observed test statistic (Wn).')
#   print('Wn measures how far your estimated information matricies deviate from equality.')
#   print('Model may not fit well if p-value less than 0.05')
#   print(test_fit_white)
# }

test_fit_kendall <- BiCopGofTest(
  u1 = u,
  u2 = v,
  family = fit$family,
  par = fit$par,
  par2 = fit$par2,
  method = 'kendall',
  B = 100 # number of bootstrap samples
)
print('Kendalls process. Returns Cramer-von Mises (CvM) and Kolmogorov-Smirnov (KS) test statistic.')
print('A small test statistic with a high p-value → fail to reject the null → model is a good fit.')
print('A large test statistic with a low p-value → reject the null → model is likely misspecified.')
print(test_fit_kendall)

#
# # Extract outputs
# fit_info <- list(
#     family = fit$famiy,
#     family_name = fit$familyname,
#     tau = fit$tau,
#     emptau = fit$emptau,
#     pvalue_indeptest = fit$p.value.indeptest,
#     lambda_U = fit$taildep$upper,
#     lambda_L = fit$taildep$lower,
#     loglik = fit$loglik,
#     effp = fit$effp,
#     AIC = fit$AIC,
#     cAIC = fit$cAIC,
#     BIC = fit$BIC,
#     pvalue_CvM = test_fit_kendall$p.value.CvM,
#     statistic_CvM = test_fit_kendall$statistic.CvM,
#     pvalue_KS = test_fit_kendall$p.value.KS,
#     statistic_KS = test_fit_kendall$statistic.KS,
#     pvalue_White = test_fit_white$p.value[1],
#     statistic_White = test_fit_white$statistic[1]
# )

# Match the copula family to the corresponding function
copula_fun <- switch(
  fit$familyname,
  "Survival Joe"    = JOcopB5,
  "Joe"             = JOcopB5,
  "Clayton"         = CLcop,
  "Survival Clayton" = CLcop,
  "Frank"           = FRcop,
  "Gaussian"        = gEVcop,
  "Gumbel"          = FGMcop,
  "Survival Gumbel" = GHcop,
  "t"               = tEVcop,
  "Independence"    = P,
  stop("Unsupported copula family: ", fit$familyname)
)

# Handle para format based on family
if (fit$familyname %in% c("Gaussian", "Frank", "Clayton", "Joe", "Gumbel", 'Survival Joe', 'Survival Gumbel')) {
  para_vec <- fit$par
} else if (fit$familyname %in% c("t", "BB1", "BB6", "BB7", "BB8")) {
  para_vec <- c(fit$par, fit$par2)
} else if (fit$familyname == "Independence") {
  para_vec <- NULL
} else {
  stop("Unsupported or unhandled copula family: ", fit$familyname)
}

# contour line at target probabilities
cdf_list <- joint.curvesCOP(
  cop = copula_fun,
  para = para_vec,
  type = "or",
  probs = target_prob,
  over.grid = True,
  delta = 0.001
)

# Extract the cDF and PDF line for the target probability
cdf_target <- cdf_list[[as.character(target_prob)]]

# Back transform the joint CDF curve to the original data scale using marginal
param_contour <- data.frame(
  x = do.call(marg_x_p$quantile, c(list(p = cdf_target$U), as.list(marg_x_p$fit$estimate))),
  y = do.call(marg_y_p$quantile, c(list(p = cdf_target$V), as.list(marg_y_p$fit$estimate)))
)

# Remove incomplete cases
param_contour <- na.omit(param_contour)


#### Plotting ####
df$RPcat <- cut(
  df[[zvar]],
  breaks = c(0, 10, 50, 100, 250, Inf),
  labels = c("<10yr", "10–50yr", "50–100yr", "100–250yr", ">250yr"),
  include.lowest = TRUE)
df <- df[order(df$RPcat), ]

cols <- c("gainsboro",
          "#bdd7e7",
          "#6baed6",
          "#2171b5",
          "#084594",
          "#021e40")


p <- ggplot() +
  #geom_contour(aes(z = return_period_emp), breaks = c(100), color = "red") +
  #geom_contour(aes(z = return_period), breaks = c(100), color = "maroon", size=1.2) +
  geom_line(data = param_contour, aes(x = x, y = y), color = "maroon", size = 1.4) +
  geom_point(data = df, aes_string(x = xvar, y = yvar, color='RPcat'), alpha = 1, size = 2.5, shape=16) +
  scale_color_manual(values = cols, name = "Return Period") +
  geom_point(data = subset(df, df[[zvar]] > 90 & df[[zvar]] < 110), aes_string(x = xvar, y = yvar)
             , shape = 21, fill = NA, color = "black", stroke = 1.2, size = 3.5, show.legend = FALSE) +
  labs(title = paste0("100-Year Joint RP Isoline (", basin, ")"),
       x = xvar, y = yvar) +
  theme_classic() +  # Adds visible axis lines
  plot_layout(guides = "collect") &
  theme(#legend.position = "bottom",
        axis.title       = element_text(size = 12, color='black'),
        axis.text        = element_text(size = 12, color='black'),
        legend.text      = element_text(size = 12, color='black'),
        legend.title     = element_text(size = 12, color='black', face='bold'),
        plot.title       = element_text(size = 12, color='black', face='bold'),
        strip.text       = element_text(size = 12, color='black'),
        #axis.line = element_line(color = "darkgrey", size = 0.5),
        panel.grid.major = element_line(color = "darkgrey", size = 0.5),  # Show major grid lines
        #panel.grid.minor = element_line(color = "grey80", size = 0.25, linetype = "dashed")
  )

ggsave(
  paste0(filename,'.png'),
  plot = p,
  width = 6,
  height = 4,
  dpi = 300
)


# #### Points above the line ####
# # Find isoline threshold (X, Y) pairs where RP = 100
# rp_target <- 100
#
# # Extract grid points near the 100-year isoline (+/- tolerance)
# tolerance <- 1  # years
# isolines_100 <- subset(grid_uv, abs(return_period - rp_target) < tolerance)
#
# # Sample the thresholds from isoline (could average or pick one)
# x_100 <- mean(isolines_100$X)
# y_100 <- mean(isolines_100$Y)
#
# # Count how many observed storms exceed BOTH thresholds
# n_above <- sum(df[[xvar]] > x_100 & df[[yvar]] > y_100)
#
# print(paste0("Points above 100-year KDE isoline: ", n_above))
#
# expected_n_above <- nrow(df) * target_exceed_prob
# print(paste0("Expected number of points above 100-year isoline: ", round(expected_n_above, 2)))
#
# cat("Actual number of exceedances: ", n_above, "\n")
# cat("Expected number of exceedances (from parametric copula): ", round(expected_n_above, 2), "\n")

sink()  # stop sinking

}