#!/bin/bash

#SBATCH --job-name=downscale_aep          # Name of the job
#SBATCH -o slurm.%j                   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                          # Total number of nodes requested (56 cores/node)
#SBATCH -t 72:00:00                          # Time limit
#SBATCH -p lowpri
#SBATCH --mem=300000

# Load necessary modules or environment (e.g., Singularity)
module purge
module load singularity/3.9.5

# Make sure to submit this script after activating the conda environment
# conda activate /home/lgrimley/miniforge3/envs/hydromt-sfincs-v1.1.0


# Call the Python script to run the scenarios for the selected storm
python /projects/sfincs/syntheticTCs_cmpdfld/MODEL_OUTPUTS/03_downscale_aep.py
