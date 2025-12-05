#!/bin/bash

# Activate environments

# source /global/homes/s/shengyu/env.sh 2pt_env

TASK=("pk") # pk, cov, win
TRACERS=("QSO") #"BGS" "LRG" "ELG_LOPnotqso" "QSO"/

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python blinded_data_pip.py 

# -----------------------------------------------
# Power spectrum and window function
# -----------------------------------------------
# splits on region and compute the power poles


if [[ " ${TASK[@]} " =~ " pk " ]]; then
    for tracer in "${TRACERS[@]}"; do
        echo ">>> Computing pk for $tracer in sky region splits:"
        srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
            python blinded_data_pip.py --version dr1-v1.5 --tracers $tracer --regions NGC SGC --todo mesh2_spectrum
    done
fi

if [[ " ${TASK[@]} " =~ " win " ]]; then
    for tracer in "${TRACERS[@]}"; do
        echo ">>> Computing window for $tracer in sky region splits:"
        srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
            python blinded_data_pip.py --version test --tracers $tracer --regions NGC SGC GCcomb --nran  --todo window_mesh2_spectrum
    done
fixw

# splits on photometric region and compute the power poles

# -----------------------------------------------
# Theoretical covariance
# -----------------------------------------------
if [[ " ${TASK[@]} " =~ " cov " ]]; then
    for tracer in "${TRACERS[@]}"; do
        echo ">>> Running $tracer for sky region splits:"
        srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
            python blinded_data_pip.py --tracers $tracer --regions NGC SGC GCcomb --zrange 0.4,0.6 --todo covariance_mesh2_spectrum
    done
fi

# srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
#     python blinded_data_pip.py --tracers LRG --regions NGC SGC --todo combine
'''