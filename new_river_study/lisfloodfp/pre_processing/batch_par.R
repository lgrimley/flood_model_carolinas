#Write .par file for LISFLOOD and run model in batch mode
#Antonia Sebastian May 11, 2020

setwd("C:/Users/anton/OneDrive/GIS/A_PROJECTS/NCPC/Onslow/NR/lisflood-fp/")

ascii <- "C:\\Users\\anton\\OneDrive\\GIS\\A_PROJECTS\\NCPC\\Onslow\\NR\\lisflood-fp\\02_input"


# Create a sequence of Manning's n values normally distributed around the values for main channels from (Chow, 1959)
#n <- rnorm(10, mean=0.05, sd=0.02)
n <- seq(from=0.03, to=0.08, by=0.005)


#WRITE THE .PAR files
for (n in n){

  writeLines(c("# New River Subset Upstream of Coastal Boundary", 
             sprintf("DEMfile     %s\\dem30m_fill.asc", ascii), 
             sprintf("dirroot     03_output\\florence\\SGCn"), 
             sprintf("resroot     florence-n%s",n),
             "sim_time            1119300",
             "initial_tstep       10",
             "massint             1000",
             "saveint             1119300",
             "threshold		        0.015",
             "infiltration		    0.0000083",
             "stagefile		        NR.stage",
             "rainfall            florence_mrms.rain",
             "bcifile			        NR_florence.bci",
             "bdyfile			        NR_florence.bdy",
             sprintf("manningfile %s\\rough_30m.asc", ascii),
             sprintf("SGCwidth		%s\\wdt30m.asc", ascii),
             sprintf("SGCbed			%s\\bed30m.asc", ascii),
             sprintf("SGCbank			%s\\bnk30m.asc", ascii),
             sprintf("SGCn			  %s", n)),
             #for each .par file, change the name of the model output (e.g., n0.03...0.08)
             sprintf("NR_florence_n%s.par",n))
  }

n <- seq(from=0.03, to=0.08, by=0.005)

#RUN THE CODE
for (n in n){
  
  shell(sprintf("lisfloodv7.exe -v NR_florence_n%s.par", n), wait=TRUE)
  
  }

