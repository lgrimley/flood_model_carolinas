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

kk=1
#for (kk in seq_along(climates)){
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
  ii=1
#  for (ii in seq_along(basins)){
    b <- basins[[ii]]
    
    print('###############################################')
    print(b)
    print('###############################################')
  
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
    target_rp <- c(25, 100)
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
    
    ##### Compute the Joint CDF using the kdecopula ####
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
    results_list[[ii]] <- as.data.frame(fit_info, stringsAsFactors = FALSE)
    
    
    #### Apply copula ####
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
    
    #### Generate probability grid #####
    
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
    fit_x <- fitdist(x, "gamma")
    fit_y <- fitdist(y, "gamma")
    u_grid <- pgamma(XY_mat$Var1, shape=fit_x$estimate["shape"], rate=fit_x$estimate["rate"])
    v_grid <- pgamma(XY_mat$Var2, shape=fit_y$estimate["shape"], rate=fit_y$estimate["rate"])
    
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
    
    #### Plotting #####
    df <- clean_data[order(clean_data$Compound_Area_sqkm_RP), ]
    df$RP <- cut(df$Compound_Area_sqkm_RP,
                 breaks = c(0, 10, 50, 100, 250, Inf),
                 labels = c("<10yr", "10-50yr", "50-100yr", "100-250yr", '>250yr'),
                 include.lowest = TRUE)
    df$RP = factor(df$RP)
    
    # Subset points where RP is around 100 (90–110)
    df_outline <- df[df$Compound_Area_sqkm_RP > 90 & df$Compound_Area_sqkm_RP < 110, ]
  
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
      
      # Add contour labels
      # geom_text_contour(data=XY_mat,
      #                   aes(x = Var1, y = Var2, z = return_period),
      #                   breaks = c(25, 99.5, 250),
      #                   label.placer = label_placer_flattest(),
      #                   size = 5, stroke = 0.2, check_overlap = TRUE,
      #                   nudge_y = -0.5, nudge_x = 3
      #                   ) +
      #annotate("text", x = 3.5, y = -0.25, label = "100-yr", size = 5, color = "black")
      
      labs(title = paste0(b,' (n=',storms_basin,')'), x = xlabel, y = ylabel) +
      theme_minimal()+ theme(text = element_text(size = 10))
    
    figname = paste0(b,'_',climate,'_KDECopula.png')
    #ggsave(figname, plot = p, width = 5, height = 3.5, dpi = 300, bg='white')
    plot_list[[ii]] <- p
    
    expected_pts = storms_basin * (1/(100 * lambda_basin))
    print(paste0('Expected poinst above 100-yr contour: ', expected_pts))
  #}
  
  #sink()
  
  combined_plot <- wrap_plots(plot_list, ncol = 2)+
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom")
  ggsave(paste0('all_basins_copula_grid_',climate,".png"), combined_plot, width = 6.5, height = 8, dpi = 300, bg = "white")
  
  results_df <- do.call(rbind, results_list)
  results_df[] <- lapply(results_df, function(x) {
    if (is.numeric(x)) round(x, 3) else x
  })
  write.csv(results_df, file = paste0('kdecopula_info_',climate,".csv"), row.names = FALSE)
  
}




