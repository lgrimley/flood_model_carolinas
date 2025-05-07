#!/bin/bash

#SBATCH --job-name=sfincs_outputs          # Name of the job
#SBATCH --output=output_%A_%a.out        # Output file for each job, %A is the job ID, %a is the array index
#SBATCH --error=error_%A_%a.err          # Error file for each job
#SBATCH --array=0-12%13                    # Job array with XXX storms, running Y concurrently
#SBATCH -t 24:00:00                          # Time limit
#SBATCH --ntasks=1                          # Only 1 task per job
#SBATCH -p lowpri
#SBATCH --mem=100000

# Load necessary modules or environment (e.g., Singularity)
module purge
module load singularity/3.9.5

# Make sure to submit this script after activating the conda environment
# conda activate /home/lgrimley/miniforge3/envs/hydromt-sfincs-v1.1.0

# Get file from list based on SLURM_ARRAY_TASK_ID
FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" input_file.txt | tr -d '\r\n')

# Call the Python script to run the scenarios for the selected storm
python /projects/sfincs/syntheticTCs_cmpdfld/MODEL_OUTPUTS/calculate_overland_depth_extent.py "$FILE"
