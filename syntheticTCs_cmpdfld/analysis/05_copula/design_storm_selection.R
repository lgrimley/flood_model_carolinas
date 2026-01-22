# Global settings
setwd('C:\\Users\\lelise\\Documents\\GitHub\\flood_model_carolinas\\syntheticTCs_cmpdfld\\analysis\\05_copula')
source("fit_marginals_and_copula.R")
source("joint_return_period_contour_v2.R")
library(ggplot2)
library(fields)
library(dplyr)
library(akima)
library(scales)

densify_rp_contour <- function(rp_contour, n_points = 500, normalize_pdf = TRUE) {
  # Densify a joint return-period contour along its arc length
  if(!all(c("x","y") %in% names(rp_contour))) {
    stop("rp_contour must contain columns 'x' and 'y'")
  }

  # Remove rows with NA in x/y/pdf
  keep_cols <- c("x","y")
  if("pdf" %in% names(rp_contour)) keep_cols <- c(keep_cols, "pdf")
  rp_contour <- rp_contour[complete.cases(rp_contour[, keep_cols]), ]

  # Compute cumulative arc length
  dx <- diff(rp_contour$x)
  dy <- diff(rp_contour$y)
  ds <- sqrt(dx^2 + dy^2)
  s <- c(0, cumsum(ds))

  s_dense <- seq(0, max(s), length.out = n_points)

  densified_x <- approx(s, rp_contour$x, xout = s_dense)$y
  densified_y <- approx(s, rp_contour$y, xout = s_dense)$y

  densified_df <- data.frame(x = densified_x, y = densified_y)

  # Interpolate pdf if present
  if("pdf" %in% names(rp_contour)) {
    densified_pdf <- approx(s, rp_contour$pdf, xout = s_dense)$y
    densified_df$pdf <- densified_pdf
    if(normalize_pdf) {
      densified_df$scaled_pdf <- (densified_df$pdf - min(densified_df$pdf)) /
        (max(densified_df$pdf) - min(densified_df$pdf))
    }
  }

  return(densified_df)
}


subset_closest_to_isoline <- function(df, rp_contour, result, xvar, yvar,
                                      n_closest = 5, prob_threshold=0.9) {

  # Ensure unique IDs
  if(!"TC_ID" %in% names(df)) df$TC_ID <- seq_len(nrow(df))

  # Scale data using the mean and stdev
  x_mean <- mean(df[[xvar]], na.rm = TRUE)
  x_sd   <- sd(df[[xvar]], na.rm = TRUE)
  y_mean <- mean(df[[yvar]], na.rm = TRUE)
  y_sd   <- sd(df[[yvar]], na.rm = TRUE)
  df_scaled <- df
  df_scaled[[xvar]] <- (df[[xvar]] - x_mean) / x_sd
  df_scaled[[yvar]] <- (df[[yvar]] - y_mean) / y_sd
  rp_contour_scaled <- rp_contour
  rp_contour_scaled$x <- (rp_contour$x - x_mean) / x_sd
  rp_contour_scaled$y <- (rp_contour$y - y_mean) / y_sd

  # Calculate the distance of each point to the isoline
  D_iso <- fields::rdist(
    as.matrix(df_scaled[, c(xvar, yvar)]),
    as.matrix(rp_contour_scaled[, c("x", "y")])
  )
  df$dist_isoline <- apply(D_iso, 1, min)

  # selec the top N closest to isoline
  df_closest <- df %>%
    dplyr::arrange(dist_isoline) %>%
    dplyr::slice(1:n_closest)

  closest_indices <- match(df_closest$TC_ID, df$TC_ID)
  nearest_idx <- apply(D_iso[closest_indices, , drop=FALSE], 1, which.min)
  df_closest$nearest_iso_ID <- nearest_idx
  df_closest$pdf <- rp_contour$pdf[nearest_idx]
  df_closest$scaled_pdf <- rp_contour$scaled_pdf[nearest_idx]
  df_closest$iso_x <- rp_contour$x[nearest_idx]
  df_closest$iso_y <- rp_contour$y[nearest_idx]

  # get the point that is the closest to the value of the marginals return period
  df_closest$return_period_x <- sapply(df_closest[[xvar]], function(x) {
    1 / (1 - do.call(result$marginals$x$cdf,
                     c(list(q = x), as.list(result$marginals$x$fit$estimate))))
  })
  df_closest$return_period_y <- sapply(df_closest[[yvar]], function(y) {
    1 / (1 - do.call(result$marginals$y$cdf,
                     c(list(q = y), as.list(result$marginals$y$fit$estimate))))
  })

  # get the point that is the closest to the joint return period
  u <- sapply(df_closest[[xvar]], function(x) do.call(result$marginals$x$cdf,
                                                      c(list(q = x), as.list(result$marginals$x$fit$estimate))))
  v <- sapply(df_closest[[yvar]], function(y) do.call(result$marginals$y$cdf,
                                                      c(list(q = y), as.list(result$marginals$y$fit$estimate))))
  joint_prob <- 1 - VineCopula::BiCopCDF(u, v,
                                         family = result$copula$family,
                                         par = result$copula$par,
                                         par2 = result$copula$par2)
  df_closest$joint_return_period <- 1 / joint_prob

  # Most likely based on combined score (distance + PDF)
  epsilon <- 1e-6
  df_closest$likelihood_score <- df_closest$scaled_pdf / (df_closest$dist_isoline + epsilon)
  most_likely_idx <- which.max(df_closest$likelihood_score)
  df_closest$most_likely_event <- FALSE
  df_closest$most_likely_event[most_likely_idx] <- TRUE

  # Closest point in the high probability region (>prob_threshold)
  high_pdf_points <- rp_contour[rp_contour$scaled_pdf >= prob_threshold, c("x","y")]
  if(nrow(high_pdf_points) > 0){
    D_highpdf <- fields::rdist(as.matrix(df_closest[, c(xvar, yvar)]),
                               as.matrix(high_pdf_points))
    closest_highpdf_idx <- apply(D_highpdf, 1, min) %>% which.min()
    df_closest$closest_to_highpdf <- FALSE
    df_closest$closest_to_highpdf[closest_highpdf_idx] <- TRUE
  } else {
    df_closest$closest_to_highpdf <- FALSE
  }

  return(df_closest)
}


datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'
outputdir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\design_events'

# Define variable pairs
variable_pairs <- list(
  list(xvar = 'stormtide', yvar = 'AvgmaxRR', zvar = 'Total_Area_sqkm_RP', climate='ncep'),
  list(xvar = 'meanMaxWS', yvar = 'MeanTotPrecipMM', zvar = 'Total_Area_sqkm_RP', climate='ncep')
)

basins <- c('CapeFear')#, 'Domain','OnslowBay','Pamlico','LowerPeeDee','Neuse')
rps <- c(100)

all_top_events <- NULL
for (basin in basins) {

  for (pair in variable_pairs) {

    xvar <- pair$xvar
    yvar <- pair$yvar
    zvar <- pair$zvar
    climate <- pair$climate

    # Load and clean data
    filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
    if (!file.exists(filepath)) next
    df <- read.csv(filepath)[, c(xvar, yvar, zvar,
                                 'Runoff_Area_sqkm_RP',
                                 'Coastal_Area_sqkm_RP',
                                 'Compound_Area_sqkm_RP')]
    df <- na.omit(df)

    # Fit marginals and copula
    x <- df[[xvar]]
    y <- df[[yvar]]
    result <- fit_marginals_and_copula(x, y)

    for (target_rp in rps){

      # Compute RP contour and PDF
      rp_contour <- joint_return_period_contour_v2(result, target_rp = target_rp,
                                                   lambda = 3.38)


      # Basic scatter + line plot
      ggplot(rp_contour, aes(x = x, y = y)) +
        geom_line(color = "blue", size = 1) +           # contour line
        geom_point(aes(color = pdf), size = 2) +        # joint PDF along contour
        scale_color_viridis_c(option = "plasma") +
        labs(
          title = "AND Joint Return-Period Contour",
          subtitle = paste0("Return period = 100 years"),
          #x = paste0("X: ", fitted$marginals$x$fit$distname),
          #y = paste0("Y: ", fitted$marginals$y$fit$distname),
          color = "Joint PDF"
        ) +
        theme_minimal(base_size = 14)


      # Densify and normalize the RP contour
      rp_contour_dense <- densify_rp_contour(rp_contour, n_points = 500,
                                             normalize_pdf = TRUE)

      # Now get the other points that are close to the line
      top_events = subset_closest_to_isoline(
        df = df,
        rp_contour = rp_contour_dense,
        result = result,
        xvar = xvar,
        yvar = yvar,
        n_closest = 10,
        prob_threshold=0.95
      )

      # Add metadata columns
      top_events$basin <- basin
      top_events$xvar <- xvar
      top_events$yvar <- yvar
      top_events$return_period <- target_rp

      # Append to the global data frame
      all_top_events <- bind_rows(all_top_events, top_events)

      # Compute distance to isoline for all points in df
      #D_iso <- rdist(as.matrix(df[, c(xvar, yvar)]), as.matrix(rp_contour_dense[, c("x", "y")]))
      #df$dist_isoline <- apply(D_iso, 1, min)

      f <- ggplot() +

        # All storm points in grey
        geom_point(data = df, aes_string(x = xvar, y = yvar), color = 'grey', alpha = 0.7) +

        # RP contour colored by scaled PDF
        geom_path(data = rp_contour_dense, aes(x = x, y = y, color = scaled_pdf), size = 1.5) +

        # Top N closest points with black border
        geom_point(data = top_events, aes_string(x = xvar, y = yvar),
                   color = "black", size = 2, stroke = 1.2) +

        # Labels for TC_ID of top events
        geom_text(data = top_events, aes_string(x = xvar, y = yvar, label = "TC_ID"),
                  vjust = -1, hjust = 0.5, size = 3.5, fontface = "bold") +

        # Highlight the most likely by distance and PDF
        # geom_point(data = top_events %>% filter(most_likely_event),
        #            aes_string(x = xvar, y = yvar),
        #            color = "purple", fill = "purple",
        #            shape = 8, size = 5, stroke = 1.5) +

        # Closest point in the high probability region (>prob_threshold)
        geom_point(data = top_events %>% filter(closest_to_highpdf),
                   aes_string(x = xvar, y = yvar),
                   color = "green4", fill = "green4",
                   shape = 17, size = 4, stroke = 1.2,
                   label='Most Likely') +

        # Divergent color scale for PDF along the contour
        scale_color_gradient2(
          low = "yellow",
          mid = "orange",
          high = "red",
          midpoint = 0.5,
          name = "Probability\nDensity"
        ) +

        theme_minimal() +
        ggtitle(paste("Basin:", basin, "| Return Period:", target_rp)) +
        theme(legend.position = "right")


        output_filepath <- file.path(outputdir, paste(basin, target_rp, xvar, yvar, ".png", sep='_'))
        ggsave(output_filepath, width = 6,height = 5, dpi = 300, bg='white')

      }
    }
}

write.csv(all_top_events, file = file.path(outputdir, "all_top_events.csv"), row.names = FALSE)


