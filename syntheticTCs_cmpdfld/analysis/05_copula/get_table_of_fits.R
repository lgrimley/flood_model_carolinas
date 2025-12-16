setwd('Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\copula_fits')
source("fit_marginals_and_copula.R")

datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'
outputdir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\copula_fits'

# Define variable pairs
variable_pairs <- list(
  list(xvar = 'stormtide', yvar = 'AvgmaxRR', zvar = 'Total_Area_sqkm_RP', climate='ncep'),
  list(xvar = 'meanMaxWS', yvar = 'MeanTotPrecipMM', zvar = 'Total_Area_sqkm_RP', climate='ncep')
)
basins <- c('Domain','OnslowBay','Pamlico', 'Neuse','LowerPeeDee','CapeFear')
all_results <- list()

for (basin in basins) {
  for (pair in variable_pairs) {

    xvar <- pair$xvar
    yvar <- pair$yvar
    zvar <- pair$zvar
    climate <- pair$climate

    # Load data
    filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
    if(!file.exists(filepath)) next
    df <- read.csv(filepath)[, c(xvar, yvar, zvar,
                                 'Runoff_Area_sqkm_RP',
                                 'Coastal_Area_sqkm_RP',
                                 'Compound_Area_sqkm_RP')]
    df <- na.omit(df)

    # Fit marginals and copula
    x <- df[[xvar]]
    y <- df[[yvar]]
    result <- fit_marginals_and_copula(x, y)
    print(result$copula$family)
    print(result$copula$familyname)

    # Get unified table
    df_all <- unify_fit_tables(result, xvar_name = xvar, yvar_name = yvar)

    # Add basin column
    df_all$basin <- basin

    # Store
    all_results[[paste(basin, xvar, yvar, sep = "_")]] <- df_all
  }
}

# Combine all results into a single table
combined_results <- do.call(rbind, all_results)


# Write to CSV
output_filepath <- file.path(outputdir, "all_basins_variable_pairs_fits.csv")
write.csv(combined_results, output_filepath, row.names = FALSE)
cat("Combined results written to:", output_filepath, "\n")
