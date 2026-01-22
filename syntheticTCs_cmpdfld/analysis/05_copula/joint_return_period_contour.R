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


make_tail_grid <- function(n_base = 300,
                           n_tail = 700,
                           tail_start = 0.9,
                           eps = 1e-6) {

  u_base <- seq(eps, tail_start, length.out = n_base)
  u_tail <- seq(tail_start, 1 - eps, length.out = n_tail)

  unique(c(u_base, u_tail))
}

solve_isoline_uv <- function(cop, target_prob,
                             u_grid,
                             eps = 1e-6,
                             tol = 1e-8) {

  # -------------------------------------------------
  # Independence copula: analytical AND isoline
  # -------------------------------------------------
  if (cop$family == 0) {

    u_valid <- u_grid[u_grid < 1 - target_prob - eps]
    v_vals  <- 1 - target_prob / (1 - u_valid)

    keep <- is.finite(v_vals) & v_vals > eps & v_vals < 1 - eps

    return(
      data.frame(
        U = u_valid[keep],
        V = v_vals[keep]
      )
    )
  }

  # -------------------------------------------------
  # General case: numerical root finding
  # -------------------------------------------------
  v_out <- rep(NA_real_, length(u_grid))

  for (i in seq_along(u_grid)) {

    u <- u_grid[i]

    f <- function(v) {

      u0 <- pmin(pmax(u, 1e-10), 1 - 1e-10)
      v0 <- pmin(pmax(v, 1e-10), 1 - 1e-10)

      1 - u0 - v0 +
        VineCopula::BiCopCDF(u0, v0,
                             family = cop$family,
                             par = cop$par,
                             par2 = cop$par2) -
        target_prob
    }


    f_lo <- f(eps)
    f_hi <- f(1 - eps)

    if (is.finite(f_lo) && is.finite(f_hi) && f_lo * f_hi < 0) {
      v_out[i] <- tryCatch(
        uniroot(f, lower = eps, upper = 1 - eps, tol = tol)$root,
        error = function(e) NA_real_
      )
    }
  }

  data.frame(U = u_grid, V = v_out) |>
    na.omit()
}




# Function to compute a joint return-period contour from a fitted copula
joint_return_period_contour <- function(fitted,
                                        target_rp = 100,
                                        lambda = 1,
                                        n_base = 300,
                                        n_tail = 700,
                                        tail_start = 0.9) {
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

  if (target_prob <= 0 || target_prob >= 1) {
    stop("Invalid target joint probability.")
  }

  # -----------------------------------------
  # Tail-adaptive copula-space isoline
  # -----------------------------------------
  u_grid <- make_tail_grid(
    n_base = n_base,
    n_tail = n_tail,
    tail_start = tail_start
  )

  # Restrict u-grid to upper tail for AND exceedance
  if (target_prob < 0.05) {
    u_grid <- u_grid[u_grid > 1 - 5 * target_prob]
  }

  contour_uv <- solve_isoline_uv(
    cop = cop,
    target_prob = target_prob,
    u_grid = u_grid
  )

  if (nrow(contour_uv) < 10) {
    # Retry with ultra-fine tail grid
    u_grid <- seq(1 - 10 * target_prob, 1 - 1e-6, length.out = 2000)
    contour_uv <- solve_isoline_uv(cop, target_prob, u_grid)
    }


  # -----------------------------------------
  # Back-transform to physical space
  # -----------------------------------------
  x_vals <- safe_quantile(
    marg_x$quantile,
    contour_uv$U,
    as.list(marg_x$fit$estimate)
  )

  y_vals <- safe_quantile(
    marg_y$quantile,
    contour_uv$V,
    as.list(marg_y$fit$estimate)
  )

  keep <- is.finite(x_vals) & is.finite(y_vals)

  x_vals <- x_vals[keep]
  y_vals <- y_vals[keep]
  U_keep <- contour_uv$U[keep]
  V_keep <- contour_uv$V[keep]

  # Enforce physical bounds if needed
  x_vals <- pmax(x_vals, 0)
  y_vals <- pmax(y_vals, 0)

  # -----------------------------------------
  # Compute joint PDF along isoline
  # -----------------------------------------
  fx <- get_marginal_density(marg_x, x_vals)
  fy <- get_marginal_density(marg_y, y_vals)

  # Final hard clipping for numerical safety
  U_keep <- pmin(pmax(U_keep, 1e-10), 1 - 1e-10)
  V_keep <- pmin(pmax(V_keep, 1e-10), 1 - 1e-10)

  cuv <- VineCopula::BiCopPDF(
    U_keep, V_keep,
    family = cop$family,
    par = cop$par,
    par2 = cop$par2
  )

  joint_pdf <- cuv * fx * fy

  rp_contour <- data.frame(
    x = x_vals,
    y = y_vals,
    pdf = joint_pdf
  )

  if (nrow(rp_contour) < 25) {
    warning("Contour has fewer than 25 points.")
  }

  return(rp_contour)
}