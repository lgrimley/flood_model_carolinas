library(VineCopula)
library(copBasic)
library(kdecopula)
library(fBasics)
library(ggplot2)
library(patchwork)
library(zoo)

fit_kde_copula <- function(df,
                           xvar,
                           yvar,
                           xvar_vec,
                           yvar_vec,
                           lambda_basin) {
  # Transform the data to pseudo observations
  u <- pobs(df[[xvar]])
  v <- pobs(df[[yvar]])
  uv <- cbind(u, v)

  # Fit the KDE copula to the pseudo obs
  emp_cop <- kdecop(udata = uv)
  summary(emp_cop)
  #plot(emp_cop, display = "3D")
  #contour(emp_cop, margins = "unif")
  # Should be close to 0
  #pkdecop(matrix(c(0, 0), ncol=2), emp_cop)
  # Should be close to 1
  #pkdecop(matrix(c(1, 1), ncol=2), emp_cop)
  # Median point should be ~0.25 if copula is close to independent
  #pkdecop(matrix(c(0.5, 0.5), ncol=2), emp_cop)

  # Extract outputs
  kdefit_info <- list(
    method = emp_cop$method,
    alpha = emp_cop$bw$alpha,
    kappa = emp_cop$bw$kappa,
    loglik = emp_cop$info$loglik,
    effp = emp_cop$info$effp,
    AIC = emp_cop$info$AIC,
    cAIC = emp_cop$info$cAIC,
    BIC = emp_cop$info$BIC
  )

  # calculate dependence measures
  dep_measures <- dep_measures(emp_cop)
  info_out <- c(kdefit_info, as.list(dep_measures))

  # Fit marginal distributions
  marg_x <- ssdFit(df[[xvar]])
  marg_y <- ssdFit(df[[yvar]])

  # Get the CDF values using smoothed spline estimate of the marginal CDF
  grid_uv <- expand.grid(X = xvar_vec, Y = yvar_vec)
  u_grid <- pssd(grid_uv$X, marg_x)
  v_grid <- pssd(grid_uv$Y, marg_y)

  # clip transformed values into (0, 1)
  epsilon <- 1e-6
  u_grid <- pmin(pmax(u_grid, epsilon), 1 - epsilon)
  v_grid <- pmin(pmax(v_grid, epsilon), 1 - epsilon)
  uv_grid <- cbind(u_grid, v_grid)

  # Get KDE copula CDF values
  kdecop_cdf <- pkdecop(uv_grid, emp_cop)
  # str(uv_grid)
  # summary(kdecop_cdf)
  # This should return true otherwise grid has values outside 0 or 1
  # all(kdecop_cdf >= 0 & kdecop_cdf <= 1)

  # Get an estimate of the joint exceedance probability
  # 1 - (P(X <= x)) - P(Y <= y) + P(X <= x, Y <=y)
  P_AND <- 1 - u_grid - v_grid + kdecop_cdf
  # Probabilities should be between 0 and 1
  P_AND <- pmax(P_AND, 0)
  grid_uv$exceed_prob = P_AND

  # Calculcate return periods from exc prob and add to grid
  RP_AND <- round(1 / (lambda_basin * P_AND), 1)
  RP_AND[is.infinite(RP_AND)] <- NA
  grid_uv$return_period <- RP_AND

  return(list(
    grid = grid_uv,
    marg_x = marg_x,
    marg_y = marg_y,
    info_out = info_out
  ))
}


fit_param_copula <- function(df,
                             xvar,
                             yvar,
                             target_prob,
                             grid_n=1000) {
  # Transform the data to pseudo observations
  u <- pobs(df[[xvar]])
  v <- pobs(df[[yvar]])
  uv <- cbind(u, v)

  ## Parametric Copula
  fit <- BiCopSelect(
    u1 = u,
    u2 = v,
    familyset = c(0,1,2,3,4,5,6),
    selectioncrit = "BIC",
    indeptest = TRUE,
    level = 0.05,
    method = "mle"
  )
  summary(fit)

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
  # pdf_target <- BiCopPDF(u1=cdf_target$U,
  #                        u2=cdf_target$V,
  #                        family=fit$family,
  #                        par=para_vec)

  return(list(
    paramCop = fit,
    cdf_target = cdf_target
  ))

  #### check ####


  # if (fit$familyname != 'Survival BB8') {
  #   # test goodness of fit of copula (VineCopula package)
  #   test_fit_white <- BiCopGofTest(
  #     u1 = u,
  #     u2 = v,
  #     family = fit$family,
  #     par = fit$par,
  #     par2 = fit$par2,
  #     method = 'white',
  #     B = 300
  #   )
  #   #print('White’s information matrix equality. Returns asymptotic p-value and observed test statistic (Wn).')
  #   #print('Wn measures how far your estimated information matricies deviate from equality.')
  #   #print('Model may not fit well if p-value less than 0.05')
  #   #print(test_fit_white)
  # }
  #
  # test_fit_kendall <- BiCopGofTest(
  #   u1 = u,
  #   u2 = v,
  #   family = fit$family,
  #   par = fit$par,
  #   par2 = fit$par2,
  #   method = 'kendall',
  #   B = 300 # number of bootstrap samples
  # )
  # #print('Kendalls process. Returns Cramer-von Mises (CvM) and Kolmogorov-Smirnov (KS) test statistic.')
  # #print('A small test statistic with a high p-value → fail to reject the null → model is a good fit.')
  # #print('A large test statistic with a low p-value → reject the null → model is likely misspecified.')
  # #print(test_fit_kendall)
  #
  #   # Extract outputs
  #   fit_info <- list(
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
  #   )

}



# Global settings
setwd(
  'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\manuscript_fig'
)
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

basins <- c('Neuse', 'OnslowBay','Pamlico','LowerPeeDee', 'Domain', 'CapeFear')
climates = c('ncep', 'canesm')
nstorms_atl = c(5018, 6200)
lambda_atl = 3.38
n_grid = 200

# Pair 1
p1 <- list(
  xvar = "stormtide",
  yvar = "AvgmaxRR",
  zvar = "Total_Area_sqkm_RP",
  epsilon_x = 0.25,
  epsilon_y = 5,
  omit_zero = TRUE
)

# Pair 2
p2 <- list(
  yvar = 'MeanTotPrecipMM',
  xvar = 'meanMaxWS',
  zvar = "Total_Area_sqkm_RP",
  epsilon_x = 10,
  epsilon_y = 50,
  omit_zero = TRUE
)
pairs <- list(p1, p2)

#for (basin in basins){
basin = 'Domain'
plots <- list()
for (p in pairs) {
  # Extract variable names and limits from pair list
  xvar <- p$xvar
  yvar <- p$yvar
  zvar <- p$zvar
  epsilon_x <- p$epsilon_x
  epsilon_y <- p$epsilon_y
  omit_zero <- p$omit_zero

  for (i in seq_along(climates)) {
    climate <- climates[i]
    nstorms <- nstorms_atl[i]

    # Read in the data and clean it
    filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
    df <- read.csv(filepath)[, c(zvar, xvar, yvar)]
    df <- na.omit(df)
    if (omit_zero == TRUE){
      df <- df[df[[zvar]] != 0 &
               df[[xvar]] != 0 &
               df[[yvar]] != 0, ]
    }

    # Calculate the lambda and target prob
    lambda_basin <- lambda_atl * (nrow(df) / nstorms)
    target_rp <- 100 # Return period  levels
    target_prob <- 1 + log(1 - 1 / target_rp) / lambda_basin
    target_exceed_prob <- 1 - target_prob

    # Setup vector limits for grid to calculate probabilities
    xvar_min = floor(min(df[[xvar]]))
    xvar_max = ceiling(max(df[[xvar]])+epsilon_x)
    yvar_min = floor(min(df[[yvar]]))
    yvar_max = ceiling(max(df[[yvar]])+epsilon_y)
    xvar_vec = seq(xvar_min, xvar_max, length.out = n_grid)
    yvar_vec = seq(yvar_min, yvar_max, length.out = n_grid)

    # Fit KDE and calculate target contour values
    print('Fitting KDE copula...')
    kde_out <- fit_kde_copula(
      df = df,
      xvar = xvar,
      yvar = yvar,
      xvar_vec = xvar_vec,
      yvar_vec = yvar_vec,
      lambda_basin = lambda_basin
    )

    #### print('Fitting parametric copula...') ####
    # # Fit parametric Copula and calculate target contour values
    # param_out <- fit_param_copula(
    #   df = df,
    #   xvar = xvar,
    #   yvar = yvar,
    #   target_prob = target_prob
    # )
    #
    # # Back transform the joint CDF curve to the original data scale using marginal
    # param_contour <- data.frame(
    #   x = qssd(param_out$cdf_target$U, kde_out$marg_x),
    #   y = qssd(param_out$cdf_target$V, kde_out$marg_y)
    # )
    #
    # param_contour <- na.approx(param_contour)

    # # Back-transform to original scale if contour was found
    # if (!is.null(param_out$cdf_target)) {
    #   param_contour <- data.frame(
    #     x = qssd(param_out$cdf_target$U, kde_out$marg_x),
    #     y = qssd(param_out$cdf_target$V, kde_out$marg_y)
    #   )
    # } else {
    #   param_contour <- NULL
    # }

    #### Plot ####
    # Categorize the points based on return period
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

    pplot <- ggplot(df, aes(x = .data[[xvar]], y = .data[[yvar]], color = RPcat)) +
      geom_point(size = 2.5, shape = 16) +
      scale_color_manual(values = cols, name = "Return Period") +
      geom_contour(
        data = kde_out$grid,
        aes(x = X, y = Y, z = exceed_prob),
        breaks = target_exceed_prob,
        color = "black",
        size=1,
        alpha=1
      ) +
      # geom_line(
      #   data = param_contour,
      #   aes(x = x, y = y),
      #   color = "red",
      #   size=1,
      #   alpha=0.75
      # ) +
      geom_point(
        data = subset(df, df[[zvar]] > 90 & df[[zvar]] < 110),
        shape = 21,
        fill = NA,
        color = "black",
        stroke = 1.2,
        size = 3.5,
        show.legend = FALSE
      ) +
      labs(title = paste(climate, "-", xvar, "&", yvar)) +
      theme_minimal()+ theme(text = element_text(size = 10))

    plots[[length(plots) + 1]] <- pplot
    cat("\nFinished:", climate, "-", xvar, "&", yvar, "\n")
  }
}

# Access plot and update axis labels
pair_map <- c(1, 1, 2, 2)
for (i in seq_along(plots)) {
  pi <- pair_map[i]
  plots[[i]] <- plots[[i]] +
    theme(legend.position = "none") +
    ggtitle(NULL)
}

plots[[1]] <- plots[[1]] +
  labs(x = "Peak Storm Tide (m)", y = "Avg Peak Rain Rate\n(mm/hr)") +
  ggtitle(paste0(basin,': Historic')) +
  xlim(0,4)+
  ylim(0,18)

plots[[2]] <- plots[[2]] +
  labs(x = "Peak Storm Tide (m)", y = " ") +
  ggtitle(paste0(basin,': Future')) +
  xlim(0,4)+
  ylim(0,18)

plots[[3]] <- plots[[3]] +
  labs(x = "Avg Peak Wind Speed (m/s)", y = "Avg Total Rainfall\n(mm)")+
  xlim(0,30)+
  ylim(0,300)

plots[[4]] <- plots[[4]] +
  labs(x = "Avg Peak Wind Speed (m/s)", y = " ") +
  xlim(0,30)+
  ylim(0,300)

# Arrange and display in 2 rows and 2 columns
final_plot <- (plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]]) +
  theme_classic() +  # Adds visible axis lines
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
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
print(final_plot)

# Save to file
ggsave(
  paste0(basin,'_Copula.png'),
  plot = final_plot,
  width = 7,
  height = 6,
  dpi = 300
)
#}
