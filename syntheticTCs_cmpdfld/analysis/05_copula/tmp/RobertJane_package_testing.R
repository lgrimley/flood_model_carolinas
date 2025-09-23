library(MultiHazard)
library(scales) #Used for transparent points
library(dplyr)  #Used for combining data frame

# Global settings
setwd(
  'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\06_copula\\manuscript_fig'
)
datadir = 'Z:\\Data-Expansion\\users\\lelise\\projects\\Carolinas_SFINCS\\Chapter3_SyntheticTCs\\05_ANALYSIS\\01_return_period_tables'

basin <- 'Neuse'
climate = 'ncep'
nstorms = 5018
lambda_atl = 3.38
n_grid = 200


# Pair 2

yvar = 'MeanTotPrecipMM'
xvar = 'meanMaxWS'
zvar = "Total_Area_sqkm_RP"

 # Read in the data and clean it
filepath <- file.path(datadir, paste0(basin, "_data_rp_", climate, ".csv"))
df <- read.csv(filepath)[, c(xvar, yvar)]
df <- na.omit(df)
df <- df[,c(2,1)]
head(df)

Rainfall.Solari<-GPD_Threshold_Solari(Event=df$MeanTotPrecipMM,
                                          Data=df$MeanTotPrecipMM)

Rainfall.Thres.Quantile<-ecdf(df$MeanTotPrecipMM)(Rainfall.Solari$Candidate_Thres)


Wind.Solari<-GPD_Threshold_Solari(Event=df$meanMaxWS,
                                  Data=df$meanMaxWS)

Wind.Thres.Quantile<-ecdf(df$meanMaxWS)(Wind.Solari$Candidate_Thres)


Diag_Non_Con(Data=df$MeanTotPrecipMM,
             Omit=,
             x_lab="MeanTotPrecipMM",
             y_lim_min=0,y_lim_max=0.03
             )

Diag_Non_Con_Sel(Data=df$MeanTotPrecipMM,
             x_lab="MeanTotPrecipMM",
             y_lim_min=0,y_lim_max=0.03,
             Selected='RGum'
             )

Diag_Non_Con(Data=df$meanMaxWS,
             Omit=,
             x_lab="meanMaxWS",
             y_lim_min=0,y_lim_max=0.15
)

Diag_Non_Con_Sel(Data=df$meanMaxWS,
                 x_lab="meanMaxWS",
                 y_lim_min=0,y_lim_max=0.15,
                 Selected='RGum'
)

# cond sampling

con.sample.Rainfall<-Con_Sampling_2D(Data_Detrend=df,
                    Data_Declust=df,
                    Con_Variable="MeanTotPrecipMM",
                    u = Rainfall.Thres.Quantile
                    )
con.sample.Rainfall

con.sample.Wind<-Con_Sampling_2D(Data_Detrend=df,
                          Data_Declust=df,
                          Con_Variable="meanMaxWS",
                          u = Wind.Thres.Quantile
                          )
con.sample.Wind

plot(con.sample.Rainfall$Data,xlim=c(0,40),ylim=c(0,300))
points(con.sample.Wind$Data,col=2)



Copula_Threshold_2D(Data_Detrend=df,
                    Data_Declust=df,
                    y_lim_min=-0.2, y_lim_max =0.5,
                    Upper=c(2,9), Lower=c(2,10),GAP=0.15
)

copula.Rain <- Copula_Threshold_2D(Data_Detrend=df,
                    Data_Declust=df,
                    u1=Rainfall.Thres.Quantile, u2=NA,
                    y_lim_min=-0.2, y_lim_max =0.5,
                    Upper=c(2,9), Lower=c(2,10),GAP=0.15
)$Copula_Family_Var1

copula.Wind <- Copula_Threshold_2D(Data_Detrend=df,
                                   Data_Declust=df,
                                   u1=NA, u2=Wind.Thres.Quantile,
                                   y_lim_min=-0.2, y_lim_max =0.5,
                                   Upper=c(2,9), Lower=c(2,10),GAP=0.15
)$Copula_Family_Var2



Bivariate<-Design_Event_2D(Data=df,
                           Data_Con1=con.sample.Rainfall$Data,
                            Data_Con2=con.sample.Wind$Data,
                            u1=Rainfall.Thres.Quantile,
                            u2=Wind.Thres.Quantile,
                            Copula_Family1=copula.Rain,
                            Copula_Family2=copula.Wind,
                          Marginal_Dist1="RGum",
                           Marginal_Dist2="RGum",
                           x_lab="MeanTotPrecipMM",y_lab="meanMaxWS",
                           RP=100,
                           N=500,
                           N_Ensemble=10
                           )