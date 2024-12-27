#!/bin/bash

#SBATCH -J test                     # Job name
#SBATCH -o slurm.%j                   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                          # Total number of nodes requested (56 cores/node)
#SBATCH --exclusive
#SBATCH -p lowpri
#SBATCH -t 72:00:00                   # Run time (hh:mm:ss) hours can be large numbers e.g., 500
#SBATCH --mem 100000

# Specify to the system to local singularity
module purge
module load singularity/3.9.5

# Download the Docker image. Will fail if .sif file already exists
# Choose your own download directory instead of /projects/sequence_analysis/vol1/prediction_work/SFINCS

#singularity pull /projects/sfincs/docker/sfincs.sif docker://deltares/sfincs-cpu:latest

## -B mounting 1 file system
time srun singularity run -B /projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/runs/TC_1525/runoff,/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/runs/TC_1525/sfincs_bc_inputs,/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/base_model /projects/sfincs/docker/sfincs.sif
