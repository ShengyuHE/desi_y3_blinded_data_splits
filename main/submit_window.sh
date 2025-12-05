#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 4  
#SBATCH -C "gpu&hbm80g"
#SBATCH --gpus-per-node=4
#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -J windows
#SBATCH -t 22:00:00 
#SBATCH -o ./logs/win_BGS-%j.out

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

srun python blinded_data_pip.py \
    --tracers BGS \
    --regions NGC SGC GCcomb \
    --nran 18 \
    --todo window_mesh2_spectrum