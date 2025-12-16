# Load required libraries for copulas, distributions, and fitting
library(VineCopula)   # For copula selection and fitting
library(copBasic)     # Basic copula functions
library(fBasics)      # Statistical functions
library(fitdistrplus) # For fitting distributions to data
library(evd)          # For Generalized Extreme Value (GEV) distributions
library(actuar)       # For Pareto distributions

# -------------------------------
# Candidate marginal distributions
# -------------------------------
# Each distribution has a CDF and quantile function
candidate_dists <- list(
  gamma   = list(cdf = pgamma, quantile = qgamma),
  lnorm   = list(cdf = plnorm, quantile = qlnorm),
  #norm    = list(cdf = pnorm, quantile = qnorm),
  weibull = list(cdf = pweibull, quantile = qweibull),
  exp     = list(cdf = pexp, quantile = qexp),
  logis   = list(cdf = plogis, quantile = qlogis),
  gev     = list(cdf = pgev, quantile = qgev),   # GEV: Gumbel/Frechet/Weibull
  gpd     = list(cdf = pgpd, quantile = qgpd),   # Generalized Pareto
  pareto  = list(cdf = ppareto, quantile = qpareto) # Pareto Type I
)

# -----------------------------------------
# Function to select the best-fit marginal
# -----------------------------------------
get_best_fit <- function(data) {
  # Remove non-finite values (NA, Inf)
  data <- data[is.finite(data)]
  data <- data[data > 0]

  if(length(data) <= 1) stop("Data must contain more than 1 finite numeric value.")

  # Container for results from all attempted fits
  all_results <- list()

  # Try fitting all candidate distributions
  fits <- lapply(names(candidate_dists), function(distname) {

    start_params <- switch(
      distname,
      gev    = list(loc = mean(data), scale = sd(data), shape = 0.1),
      gpd    = list(loc = min(data), scale = sd(data), shape = 0.1),
      pareto = list(shape = 1, scale = min(data)),
      NULL
    )

    fit_obj <- tryCatch(
      suppressWarnings(fitdist(data, distname, start = start_params)),
      error = function(e) e
    )

    # Record this attempt
    if (inherits(fit_obj, "error")) {
      all_results[[distname]] <<- data.frame(
        dist = distname,
        aic = NA,
        bic = NA,
        loglik = NA,
        params = NA,
        status = paste("FAILED:", fit_obj$message),
        stringsAsFactors = FALSE
      )
      return(NULL)
    } else {
      all_results[[distname]] <<- data.frame(
        dist = distname,
        aic = fit_obj$aic,
        bic = fit_obj$bic,
        loglik = fit_obj$loglik,
        params = paste(
          names(fit_obj$estimate),
          round(fit_obj$estimate, 5),
          collapse = "; "
        ),
        status = "OK",
        stringsAsFactors = FALSE
      )
      return(fit_obj)
    }
  })

  # Remove failed fits
  fits <- Filter(Negate(is.null), fits)

  # Full results table for ALL distributions
  all_fits_df <- do.call(rbind, all_results)

  # Select the distribution with lowest AIC
  best_fit_index <- which.min(sapply(fits, function(f) f$aic))
  best_fit <- fits[[best_fit_index]]
  best_name <- best_fit$distname

  # Add a logical column indicating the best fit
  all_fits_df$best <- all_fits_df$dist == best_name

  # Return original structure, plus all_fits
  list(
    name = best_name,
    fit = best_fit,
    cdf = candidate_dists[[best_name]]$cdf,
    quantile = candidate_dists[[best_name]]$quantile,
    all_fits = all_fits_df   # full table of all attempts with best column
  )
}


# ---------------------------------------------------
# Function to fit marginals and a copula for x and y
# ---------------------------------------------------
fit_marginals_and_copula <- function(x, y, familyset = c(0,1,2,3,4,5,6), indeptest = TRUE) {

  # Ensure x and y are same length
  if(length(x) != length(y)) stop("x and y must have the same length.")

  # ----------------
  # Fit Marginals
  # ----------------
  marg_x <- get_best_fit(x)
  marg_y <- get_best_fit(y)

  # ----------------------------
  # Convert to pseudo-observations
  # ----------------------------
  u <- pobs(x)
  v <- pobs(y)

  # ----------------------------
  # Fit best copula (existing behavior)
  # ----------------------------
  best_cop <- BiCopSelect(
    u1 = u,
    u2 = v,
    familyset = familyset,
    selectioncrit = "AIC",
    indeptest = indeptest,
    method = "mle"
  )

  # -----------------------------------------
  # New: Evaluate ALL copulas in the familyset
  # -----------------------------------------
  copula_results <- lapply(familyset, function(fam) {
    tryCatch({
      fit <- BiCopEst(u, v, family = fam, method = "mle")

      data.frame(
        family_num = fam,
        family = BiCopName(fam),
        aic = fit$AIC,
        bic = fit$BIC,
        loglik = fit$logLik,
        tau = BiCopPar2Tau(fit$family, fit$par, fit$par2),
        params = paste(
          c("par", "par2")[1:length(na.omit(c(fit$par, fit$par2)))],
          round(na.omit(c(fit$par, fit$par2)), 5),
          collapse = "; "
        ),
        status = "OK",
        stringsAsFactors = FALSE
      )
    },
    error = function(e) {
      data.frame(
        family_num = fam,
        family = BiCopName(fam),
        aic = NA,
        bic = NA,
        loglik = NA,
        tau = NA,
        params = NA,
        status = paste("FAILED:", e$message),
        stringsAsFactors = FALSE
      )
    })
  })

  copula_df <- do.call(rbind, copula_results)

  # -----------------------------------------
  # Return structure (extend, don’t break)
  # -----------------------------------------
  list(
    marginals = list(
      x = marg_x,
      y = marg_y
    ),
    copula = best_cop,      # original behavior
    copula_all = copula_df  # NEW: table of all copulas
  )
}


# ---------------------------------------------------
# Function to combine all info on tested fits
# ---------------------------------------------------
unify_fit_tables <- function(result, xvar_name = "x", yvar_name = "y", top_n = 3, digits = 2) {

  # -----------------------
  # Prepare marginal tables
  # -----------------------
  xdf <- result$marginals$x$all_fits
  ydf <- result$marginals$y$all_fits

  if("dist" %in% names(xdf)) names(xdf)[names(xdf) == "dist"] <- "model"
  if("dist" %in% names(ydf)) names(ydf)[names(ydf) == "dist"] <- "model"

  for(col in c("tau","xvar","yvar","best")) {
    if(!col %in% names(xdf)) xdf[[col]] <- NA
    if(!col %in% names(ydf)) ydf[[col]] <- NA
  }

  xdf$variable <- xvar_name
  xdf$model_type <- "marginal"
  xdf$xvar <- xvar_name
  xdf$yvar <- yvar_name
  xdf$family_num_explicit <- NA
  xdf$selected_by_BiCopSelect <- FALSE

  ydf$variable <- yvar_name
  ydf$model_type <- "marginal"
  ydf$xvar <- xvar_name
  ydf$yvar <- yvar_name
  ydf$family_num_explicit <- NA
  ydf$selected_by_BiCopSelect <- FALSE

  # -----------------------
  # Prepare copula table
  # -----------------------
  cdf <- result$copula_all

  for(col in c("params","best")) {
    if(!col %in% names(cdf)) cdf[[col]] <- NA
  }

  cdf$family_num_explicit <- cdf$family_num
  cdf$model <- BiCopName(cdf$family_num)
  cdf$variable <- paste(xvar_name, "+", yvar_name)
  cdf$model_type <- "copula"
  cdf$xvar <- xvar_name
  cdf$yvar <- yvar_name

  # -----------------------
  # Add a row for BiCopSelect() result
  # -----------------------
  cop_select <- result$copula
  copula_select_row <- data.frame(
    family_num = cop_select$family,
    family = BiCopName(cop_select$family),
    aic = cop_select$AIC,
    bic = cop_select$BIC,
    loglik = cop_select$logLik,
    tau = BiCopPar2Tau(cop_select$family, cop_select$par, cop_select$par2),
    params = paste(
      c("par","par2")[1:length(na.omit(c(cop_select$par, cop_select$par2)))],
      round(na.omit(c(cop_select$par, cop_select$par2)),5),
      collapse = "; "
    ),
    status = "BiCopSelect",
    best = TRUE,
    family_num_explicit = cop_select$family,
    model = BiCopName(cop_select$familyname),
    variable = paste(xvar_name, "+", yvar_name),
    model_type = "copula",
    xvar = xvar_name,
    yvar = yvar_name,
    stringsAsFactors = FALSE
  )

  # Ensure all columns exist in copula_select_row
  missing_cols <- setdiff(names(cdf), names(copula_select_row))
  for(col in missing_cols) copula_select_row[[col]] <- NA

  # Reorder columns to match cdf
  copula_select_row <- copula_select_row[, names(cdf)]

  # Append BiCopSelect row to copula table
  cdf <- rbind(cdf, copula_select_row)

  # Ensure all required columns exist in each table
  ensure_cols <- function(df, cols) {
    missing <- setdiff(cols, names(df))
    for(col in missing) df[[col]] <- NA
    df[, cols, drop = FALSE]  # reorder columns
  }

  # -----------------------
  # Combine all tables
  # -----------------------
  required_cols <- c("variable","model_type","model","family_num_explicit","aic","bic","loglik",
                     "tau","params","status","xvar","yvar","best")

  combined <- rbind(
    xdf[, intersect(required_cols, names(xdf))],
    ydf[, intersect(required_cols, names(ydf))],
    cdf[, required_cols]
  )

  # -----------------------
  # Round numeric columns
  # -----------------------
  num_cols <- c("aic","bic","loglik","tau")
  combined[num_cols] <- lapply(combined[num_cols], function(x) ifelse(!is.na(x), round(x, digits), NA))

  # -----------------------
  # Keep top N models by AIC for each variable + model_type, but always include BiCopSelect
  # -----------------------
  combined <- do.call(rbind, lapply(split(combined, paste(combined$variable, combined$model_type)), function(df) {
    top_df <- df[order(df$aic, na.last = NA), ]
    # Ensure BiCopSelect row is included even if not in top N
    if(any(df$selected_by_BiCopSelect)) {
      select_row <- df[df$selected_by_BiCopSelect, ]
      top_df <- rbind(select_row, top_df)
      top_df <- top_df[!duplicated(top_df$model), ]  # avoid duplicate row if BiCopSelect already in top N
    }
    head(top_df, max(top_n, 1))
  }))

  rownames(combined) <- NULL
  combined
}
