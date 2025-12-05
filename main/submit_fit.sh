#!/bin/bash
#SBATCH -J fit
#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus 4
#SBATCH -n 4
#SBATCH -o slurm-%x-%j.out
#SBATCH --array=0-1     # 0 = FM, 1 = SF

# Load environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# source /global/homes/s/shengyu/env.sh fit_env

# Tracers
TRACERS=("BGS" "LRG" "ELG" "QSO")

# Approaches controlled by the job array
APPROACHES=("FM" "SF")
approach=${APPROACHES[$SLURM_ARRAY_TASK_ID]}

echo ">>> SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo ">>> Approach: $approach"
echo ">>> Tracers: ${TRACERS[@]}"

for tracer in "${TRACERS[@]}"; do
    echo ">>> Running tracer: $tracer, approach: $approach"
    srun python fit_blinded_data.py --tracers "$tracer" --regions NGC SGC GCcomb --approaches "$approach"
done

echo ">>> All tracers completed for approach: $approach"