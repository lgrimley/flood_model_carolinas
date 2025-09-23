library(VineCopula)
library(copBasic)
library(fBasics)
library(ggplot2)
library(patchwork)
library(zoo)
library(fitdistrplus)

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

# Global settings
setwd(
  'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\ncep_parametric_copulas'
)

datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

basins <- c('Neuse', 'OnslowBay','Pamlico','LowerPeeDee', 'Domain', 'CapeFear')
climate = 'ncep'
nstorms = 5018
yvar = 'MeanTotPrecipMM'
xvar = 'AvgmaxRR'
zvar = "Runoff_Area_sqkm_RP"
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


  # Function to fit all and return the best fit by AIC
  # Candidate distributions and their corresponding CDF and quantile function names
  marg_x_p <- get_best_fit(df[[xvar]])
  marg_y_p <- get_best_fit(df[[yvar]])
  cat("Best-fit distribution for", xvar, ":", marg_x_p$name, "\n")
  cat("Best-fit distribution for", yvar, ":", marg_y_p$name, "\n")

  print(marg_x_p)
  print(marg_y_p)

  # Get best fit copula
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

  # Goodness of fit test
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

  sink()

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


}