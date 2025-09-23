library(VineCopula); library(copBasic); library(kdecopula)
library(fBasics); library(ggplot2); library(patchwork)

run_analysis <- function(basin, climate, datadir, storms_total, lambda_atl) {
  fn <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
  df <- na.omit(read.csv(fn)[, c("Compound_Area_sqkm_RP", xvar, yvar)])
  df <- df[df[[xvar]] != 0, ]
  nstorms <- nrow(df); lambda_basin <- lambda_atl * (nstorms / storms_total)
  u <- pobs(df[[xvar]]); v <- pobs(df[[yvar]]); uv <- cbind(u, v)
  
  ## Tail dependence
  alpha <- 0.1
  lambda_U <- mean(v[u > 1 - alpha] > 1 - alpha)
  lambda_L <- mean(v[u < alpha] < alpha)
  
  ## KDE Copula
  emp_cop <- kdecop(udata = uv)
  grid_uv <- expand.grid(X = seq(min(df[[xvar]]), max(df[[xvar]]), len = 200),
                         Y = seq(min(df[[yvar]]), max(df[[yvar]]), len = 200))
  marg_x <- ssdFit(df[[xvar]]); marg_y <- ssdFit(df[[yvar]])
  u_g <- pssd(grid_uv$X, marg_x); v_g <- pssd(grid_uv$Y, marg_y)
  pr_and <- pmax(1 - u_g - v_g + pkdecop(cbind(u_g, v_g), emp_cop), 0)
  grid_uv$RP <- round(1 / (lambda_basin * pr_and), 1)
  
  ## Parametric Copula
  fit <- BiCopSelect(u1 = u, u2 = v, familyset = NA,
                     selectioncrit = "BIC", indeptest = TRUE, method = "mle")
  # Match the copula family to the corresponding function
  copula_fun <- switch(
    fit$familyname,
    "Survival Joe"    = JOcopB5,
    "Joe"             = JOcopB5,
    "Clayton"         = CLcop,
    "Survival Clayton"= CLcop,
    "Frank"           = FRcop,
    "Gaussian"        = gEVcop,
    "Gumbel"          = GHcop,
    "t"               = tEVcop,
    stop("Unsupported copula family: ", fit$familyname)
  )
  
  # contour line at 100‑yr
  pr100 <- 1 / (100 * lambda_basin)
  cdf_list <- joint.curvesCOP(cop = copula_fun,
                              para = c(fit$par, fit$par2), 
                              type = "or", 
                              probs = target_prob,
                              over.grid = True,
                              delta=0.001
                              )
  
  # Extract the line for pr100
  cdf_line <- cdf_list[[as.character(pr100)]]
  
  df_line <- data.frame(
    x = qssd(cdf_line$U, marg_x),
    y = qssd(cdf_line$V, marg_y)
  )
  
  ## Plot
  df$RPcat <- cut(df$Compound_Area_sqkm_RP,
                  breaks = c(0,10,50,100,250, Inf),
                  labels = c("<10yr","10–50yr","50–100yr","100–250yr",">250yr"))
  cols <- c("gainsboro", "#bdd7e7","#6baed6","#2171b5","#084594","#021e40")
  
  ggplot(df, aes(x = .data[[xvar]], y = .data[[yvar]], color = RPcat)) +
    geom_point(size = 2.5, shape = 16) +
    scale_color_manual(values = cols, name = "Return Period") +
    geom_contour(data = grid_uv, aes(x = X, y = Y, z = RP),
                 breaks = 100, color = "black", linetype = "longdash") +
    geom_line(data = df_line, aes(x = x, y = y),
              color = "red", linetype = "longdash") +
    geom_point(data = subset(df, Compound_Area_sqkm_RP > 90 & Compound_Area_sqkm_RP < 110),
               shape = 21, fill = NA, color = "black", stroke = 1.2, size = 3.5,
               show.legend = FALSE) +
    labs(
      title = paste0(basin, " (n=", nstorms, ")"),
      x = xlabel, y = ylabel
    ) +
    theme_minimal(base_size = 10)
}

# Global settings
setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\manuscript_fig')
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

basins <- c("CapeFear","Pamlico","Neuse","OnslowBay","LowerPeeDee","Domain")
climates <- c("ncep","canesm")
nstorms_totals <- c(5018, 6200)
lambda_atlantic <- 3.38
xvar <- "stormtide"
yvar <- "AvgmaxRR"
xlabel <- "Storm Tide (m)"
ylabel <- "Avg Peak Rain Rate\n(mm/hr)"
target_rp <- c(10,25,50,100,250,500,1000)
target_prob <- NULL

test = run_analysis(b=basins[1], climates[1], datadir, nstorms_totals[1], lambda_atlantic)

#plots <- list()
#for (b in basins) {
#  plots[[paste0(b, "_past")]]   <- run_analysis(b, climates[1], datadir, nstorms_totals[1], lambda_atlantic)
#  plots[[paste0(b, "_future")]] <- run_analysis(b, climates[2], datadir, nstorms_totals[2], lambda_atlantic)
#}

# Combine last basin example
p1 <- plots[[1]]; p2 <- plots[[2]]
combined <- p1 + theme(legend.position = "none") | p2
combined <- combined + plot_layout(guides = "collect") & theme(legend.position = "bottom")
#ggsave("example.png", plot = combined, width = 6.5, height = 6, dpi = 300)
