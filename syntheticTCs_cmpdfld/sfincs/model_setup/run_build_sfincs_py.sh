#!/bin/bash -l

#SBATCH -J mod_buil   # job name
#SBATCH -o slurm.%j
#SBATCH -N 1          # number of nodes requested
#SBATCH -p lowpri     # queue
#SBATCH -t 36:00:00   # job time allowed
#SBATCH --mem 500000  # memory requested

module purge

# Activate the python environment
conda activate hydromt-sfincs-v1.1.0

# Run the python script and output to slurm script
time srun python -u /projects/sfincs/build_sfincs_ENC_200m_sbg5m_Ch3.py
