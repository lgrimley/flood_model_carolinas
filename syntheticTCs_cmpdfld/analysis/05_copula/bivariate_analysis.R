# Global settings
setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\copula_fits')
source("fit_marginals_and_copula.R")
source("joint_return_period_contour.R")
library(ggplot2)
library(fields) # rdist for pairwise Euclidean distances
library(akima)
library(patchwork)

# Function to densify a 2D contour along its curve length
densify_contour <- function(contour_df, n_points = 200) {
  # contour_df: data.frame with columns x and y
  # n_points : desired number of points along the curve

  # 1. Compute distances between consecutive points
  dx <- diff(contour_df$x)
  dy <- diff(contour_df$y)
  seg_lengths <- sqrt(dx^2 + dy^2)

  # 2. Cumulative arc length
  arc_len <- c(0, cumsum(seg_lengths))

  # 3. Generate evenly spaced points along the curve length
  arc_len_dense <- seq(0, max(arc_len), length.out = n_points)

  # 4. Interpolate x and y along arc length
  x_dense <- approx(arc_len, contour_df$x, xout = arc_len_dense)$y
  y_dense <- approx(arc_len, contour_df$y, xout = arc_len_dense)$y

  # 5. Return densified contour
  densified_df <- data.frame(x = x_dense, y = y_dense)
  return(densified_df)
}


datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'
outputdir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\copula_fits'

# Define variable pairs
variable_pairs <- list(
  list(xvar = 'stormtide', yvar = 'AvgmaxRR', zvar = 'Total_Area_sqkm_RP', climate='ncep'),
  list(xvar = 'meanMaxWS', yvar = 'MeanTotPrecipMM', zvar = 'Total_Area_sqkm_RP', climate='ncep')
)

basins <- c('Neuse', 'Pamlico')#, c('OnslowBay','Pamlico') #c('LowerPeeDee','CapeFear')

plots_meta <- list()
plots <- list()
plot_idx <- 1

for (basin in basins) {
  for (pair in variable_pairs) {

    xvar <- pair$xvar
    yvar <- pair$yvar
    zvar <- pair$zvar
    climate <- pair$climate

    filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
    if (!file.exists(filepath)) next
    df <- read.csv(filepath)[, c(xvar, yvar, zvar,
                                 'Runoff_Area_sqkm_RP',
                                 'Coastal_Area_sqkm_RP',
                                 'Compound_Area_sqkm_RP')]
    df <- na.omit(df)

    # RP categories
    df$RPcat <- cut(df[[zvar]],
                    breaks = c(0, 10, 50, 100, 250, Inf),
                    labels = c("<10yr", "10–50yr", "50–100yr", "100–250yr", ">250yr"),
                    include.lowest = TRUE)
    df <- df[order(df$RPcat), ]
    cols <- c("gainsboro", "#bdd7e7", "#6baed6", "#2171b5", "#084594", "#021e40")

    # Fit marginals and copula
    x <- df[[xvar]]
    y <- df[[yvar]]
    result <- fit_marginals_and_copula(x, y)

    # Compute 100-yr RP contour
    rp_contour <- joint_return_period_contour(result, target_rp = 100, lambda = 3.38)
    rp_contour_dense <- densify_contour(rp_contour, n_points = 400)

    # Plot
    p <- ggplot() +
      # Points colored by RP category
      geom_point(data = df, aes_string(x = xvar, y = yvar, color = "RPcat"),
                 alpha = 1, size = 2.5, shape = 16) +
      scale_color_manual(values = cols, name = "Return Period") +

      # 100-year RP contour line ONLY (no extra points)
      geom_path(data = rp_contour_dense, aes(x = x, y = y),
                color = "black", size = 1.4, alpha = 0.75) +

      # Highlight special points exactly as before
      geom_point(data = subset(df, df[[zvar]] > 90 & df[[zvar]] < 110),
                 aes_string(x = xvar, y = yvar), shape = 22, fill = NA,
                 color = "black", stroke = 1.8, size = 3.7, alpha = 0.8) +
      geom_point(data = subset(df, df[['Runoff_Area_sqkm_RP']] > 90 & df[['Runoff_Area_sqkm_RP']] < 110),
                 aes_string(x = xvar, y = yvar), shape = 23, fill = NA,
                 color = "maroon", stroke = 1.8, size = 3.5, alpha = 0.7) +
      geom_point(data = subset(df, df[['Coastal_Area_sqkm_RP']] > 90 & df[['Coastal_Area_sqkm_RP']] < 110),
                 aes_string(x = xvar, y = yvar), shape = 24, fill = NA,
                 color = "seagreen", stroke = 1.8, size = 3.5, alpha = 0.7) +
      geom_point(data = subset(df, df[['Compound_Area_sqkm_RP']] > 90 & df[['Compound_Area_sqkm_RP']] < 110),
                 aes_string(x = xvar, y = yvar), shape = 21, fill = NA,
                 color = "orange", stroke = 1.8, size = 3.5, alpha = 0.8) +

      # Labels and theme
      labs(title = basin) +
      theme_classic() +
      theme(axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            legend.text = element_text(size = 10),
            legend.title = element_text(size = 12, face = "bold"))

    # Store plot and metadata
    plots_meta[[plot_idx]] <- list(plot = p, basin = basin, xvar = xvar, yvar = yvar)
    plot_idx <- plot_idx + 1
  }
}


for (i in seq_along(plots_meta)) {
  p_info <- plots_meta[[i]]
  # Customize labels according to basin or variable
  xlab <- ifelse(p_info$xvar == "stormtide", "Peak Storm Tide (m)", "Avg Peak Wind Speed (m/s)")
  ylab <- ifelse(p_info$yvar == "AvgmaxRR", "Avg Peak Rain Rate\n(mm/hr)", "Avg Total Rainfall\n(mm)")

  plots_meta[[i]]$plot <- p_info$plot +
    labs(x = xlab, y = ylab, title = p_info$basin) +
    theme(plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_line(color = "grey80"))
}

final_plot <- (plots_meta[[1]]$plot | plots_meta[[3]]$plot) /
  (plots_meta[[2]]$plot | plots_meta[[4]]$plot) +
  theme_classic() +  # Adds visible axis lines
  plot_layout(guides = "collect") &
  theme(legend.position = "right",
        axis.title       = element_text(size = 10, color='black'),
        axis.text        = element_text(size = 10, color='black'),
        legend.text      = element_text(size = 10, color='black'),
        legend.title     = element_text(size = 10, color='black', face='bold'),
        plot.title       = element_text(size = 10, color='black', face='bold'),
        strip.text       = element_text(size = 10, color='black'),
        #axis.line = element_line(color = "darkgrey", size = 0.5),
        panel.grid.major = element_line(color = "darkgrey", size = 0.5),  # Show major grid lines
        #panel.grid.minor = element_line(color = "grey80", size = 0.25, linetype = "dashed")
  )
print(final_plot)

output_filepath <- file.path(outputdir, "Neuse_Pamlico_copula.png")
ggsave(
  output_filepath,
  plot = final_plot,
  width = 6.5,
  height = 5,
  dpi = 300
)




# raw df points
#points_df <- df[, c(xvar, yvar)]
# contour points
#points_contour <- rp_contour_dense[, c("x", "y")]
# Compute full pairwise distance matrix
#dist_mat <- rdist(as.matrix(points_df), as.matrix(points_contour))
# Minimum distance to contour for each df point
#min_dist <- apply(dist_mat, 1, min)
# Indices of the 10 closest points
#top10_idx <- order(min_dist)[1:1]
# Subset the closest 10 points
#closest_points <- df[top10_idx, ]
# maximum density for Adam P
# 5 - 500yr




