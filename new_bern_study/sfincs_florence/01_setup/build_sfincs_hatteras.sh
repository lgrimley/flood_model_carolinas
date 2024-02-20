#!/bin/bash -l

#SBATCH -J mod_buil
#SBATCH -o slurm.%j
#SBATCH -N 1
#SBATCH -p lowpri
#SBATCH -t 36:00:00
#SBATCH --mem 400000

module purge
#module load python

conda activate hydromt-sfincs-dev

time srun python -u /projects/sfincs/update_soil_sfincs_nbll.py
