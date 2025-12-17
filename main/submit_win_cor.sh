#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4  
#SBATCH -C "gpu&hbm80g"
#SBATCH --gpus-per-node=4
#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -J covariance
#SBATCH -t 41:00:00
#SBATCH -o ./logs/cor_SnoDES-%j.out

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

region=SnoDES # N, S, noDES, SnoDES
# for indx in {0..6}; do
#     srun -N 1 -n 4 python blinded_data_pip.py --indx $indx --regions $region --todo window_mesh2_spectrum
# done
for indx in {0..6}; do
    srun -N 1 -n 4 python blinded_data_pip.py --indx $indx --regions $region --todo covariance_mesh2_spectrum
done