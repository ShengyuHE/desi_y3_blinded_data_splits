#!/bin/bash

# Activate environments
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

# gpu enviroment for 2pt (use jax own CUDA)
# source /global/homes/s/shengyu/env.sh fit_env

TRACERS=("ELG") #"BGS" "LRG" "ELG" "QSO"

# -----------------------------------------------
# Cosmological fitting
# -----------------------------------------------
# splits on region and compute the power poles
echo ">>> Running $tracer for fitting:"
for tracer in "${TRACERS[@]}"; do
    srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
        python fit_blinded_data.py --tracers $tracer --regions NGC SGC GCcomb --approaches FM
    # srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g \
        # python fit_blinded_data.py --tracers $tracer --regions NGC SGC GCcomb --approaches SF 
done
