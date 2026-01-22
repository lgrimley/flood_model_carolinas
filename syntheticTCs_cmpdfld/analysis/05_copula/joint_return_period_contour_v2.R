# Load required libraries for copulas, distributions, and fitting
library(VineCopula)   # For copula selection and fitting
library(copBasic)     # Basic copula functions
library(fBasics)      # Statistical functions
library(fitdistrplus) # For fitting distributions to data
library(evd)          # For Generalized Extreme Value (GEV) distributions
library(actuar)       # For Pareto distributions


get_marginal_density <- function(marginal, x) {
  dist_name <- marginal$fit$distname
  params <- as.list(marginal$fit$estimate)
  dens_fun <- match.fun(paste0("d", dist_name))
  do.call(dens_fun, c(list(x = x), params))
}

safe_quantile <- function(qfun, p, params,
                          pmin = 1e-4,
                          pmax = 1 - 1e-4) {
  # constrain probabilities and flag invalid results
  p <- pmin(pmax(p, pmin), pmax)

  q <- suppressWarnings(
    do.call(qfun, c(list(p = p), params))
  )

  q[!is.finite(q)] <- NA
  q
}


# Function to compute a joint return-period contour from a fitted copula
joint_return_period_contour_v2 <- function(fitted, target_rp = 100, lambda = 1) {
  # Inputs:
  #   fitted    : result from fit_marginals_and_copula(x, y)
  #   target_rp : desired return period in years (e.g., 100)
  #   lambda    : average annual storm rate (for scaling, optional)
  # Outputs:
  #   data.frame with x and y values for the isoline


  # Extract fitted marginals and copula
  marg_x <- fitted$marginals$x
  marg_y <- fitted$marginals$y
  cop    <- fitted$copula

  # -----------------------------------------
  # Compute target joint exceedance probability
  # Adjust for annual rate if provided
  # -----------------------------------------
  target_prob <- 1 + log(1 - 1 / target_rp) / lambda

  # -------------------------------------------------------------------
  # Vectorized solution along fine u grid
  # -------------------------------------------------------------------
  n_grid <- 2000
  u_grid <- seq(1e-6, 1-1e-6, length.out = n_grid)

  solve_v_for_u <- function(u, cop, p_target){
    # Solve 1 - u - v + C(u,v) = p_target
    f <- function(v) 1 - u - v + VineCopula::BiCopCDF(u, v, family=cop$family, par=cop$par, par2=cop$par2) - p_target
    # v must be between u and 1
    #tryCatch(uniroot(f, interval = c(u, 1-1e-6))$root, error=function(e) NA)
  }

  v_grid <- sapply(u_grid, solve_v_for_u, cop=cop, p_target=target_prob)


  # -------------------------------------------------------------------
  # Transform back to physical space
  # -------------------------------------------------------------------
  uv$x <- safe_quantile(marg_x$quantile, uv$U, as.list(marg_x$fit$estimate))
  uv$y <- safe_quantile(marg_y$quantile, uv$V, as.list(marg_y$fit$estimate))


  # -------------------------------------------------------------------
  # Compute joint PDF along the contour
  # -------------------------------------------------------------------
  fx <- get_marginal_density(marg_x, uv$x)
  fy <- get_marginal_density(marg_y, uv$y)
  cuv <- VineCopula::BiCopPDF(uv$U, uv$V, family = cop$family, par = cop$par, par2 = cop$par2)
  uv$pdf <- fx * fy * cuv

  rp_contour <- data.frame(x = uv$x, y = uv$y, pdf = uv$pdf)

  # Ensure at least 25 points
  if (nrow(rp_contour) < 25) {
    warning("Contour has fewer than 25 points. Consider reducing delta in joint.curvesCOP or checking data.")
  }

  return(rp_contour)
}