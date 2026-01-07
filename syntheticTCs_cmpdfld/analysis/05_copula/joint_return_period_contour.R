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
joint_return_period_contour <- function(fitted, target_rp = 100, lambda = 1) {
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

  # -----------------------------------------
  # Match the copula family to the corresponding function
  # (same as in your original script)
  # -----------------------------------------
  copula_fun <- switch(
    cop$familyname,
    "Survival Joe"     = JOcopB5,
    "Joe"              = JOcopB5,
    "Clayton"          = CLcop,
    "Survival Clayton" = CLcop,
    "Frank"            = FRcop,
    "Gaussian"         = gEVcop,
    "Gumbel"           = FGMcop,
    "Survival Gumbel"  = GHcop,
    "t"                = tEVcop,
    "Independence"     = P,
    stop("Unsupported copula family: ", cop$familyname)
  )

  # Handle para format based on family
  para_vec <- switch(
    cop$familyname,
    "Gaussian" = cop$par,
    "Frank"    = cop$par,
    "Clayton"  = cop$par,
    "Joe"      = cop$par,
    "Gumbel"   = cop$par,
    'Survival Joe'    = cop$par,
    'Survival Gumbel' = cop$par,
    "t"        = c(cop$par, cop$par2),
    "Independence" = NULL
  )

  # -----------------------------------------
  # Generate the curve in copula space using joint.curvesCOP
  # over a fine grid (guarantees many points)
  # -----------------------------------------
  cdf_list <- joint.curvesCOP(
    cop = copula_fun,
    para = para_vec,
    type = "or",
    probs = target_prob,
    over.grid = TRUE,
    delta = 0.0001
  )

  contour_uv <- cdf_list[[as.character(target_prob)]]

  # Clip U,V to avoid p=0 or 1
  contour_uv$U <- pmin(pmax(contour_uv$U, 1e-6), 1 - 1e-6)
  contour_uv$V <- pmin(pmax(contour_uv$V, 1e-6), 1 - 1e-6)

  # Transform back to original data
  # Note: For GEVs bounds depend on fitted parameters and if the quantile
  # is above this bound it will be undefined (NA)
  x_vals <- safe_quantile(marg_x$quantile, contour_uv$U, as.list(marg_x$fit$estimate))
  y_vals <- safe_quantile(marg_y$quantile, contour_uv$V, as.list(marg_y$fit$estimate))

  # Drop invalid isoline points (not physically realizable under the
  # fitted marginals) before calculating densities
  keep <- is.finite(x_vals) & is.finite(y_vals)
  x_vals <- x_vals[keep]
  y_vals <- y_vals[keep]
  U_keep <- contour_uv$U[keep]
  V_keep <- contour_uv$V[keep]

  # Clip physically impossible
  x_vals <- pmax(x_vals, 0)
  y_vals <- pmax(y_vals, 0)

  # Compute marginal densities
  fx <- get_marginal_density(marg_x, x_vals)
  fy <- get_marginal_density(marg_y, y_vals)

  # Compute copula density (output is in copula space)
  cuv <- VineCopula::BiCopPDF(U_keep, V_keep, family = cop$family,
                            par = cop$par, par2 = cop$par2)

  # Joint PDF (in physical space) = copula density * marginal densities
  joint_pdf <- cuv * fx * fy

  rp_contour <- data.frame(x = x_vals, y = y_vals, pdf = joint_pdf)

  # Ensure at least 25 points
  if (nrow(rp_contour) < 25) {
    warning("Contour has fewer than 25 points. Consider reducing delta in joint.curvesCOP or checking data.")
  }

  return(rp_contour)
}