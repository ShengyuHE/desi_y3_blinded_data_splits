#!/bin/bash

# !!! Process one catalog bin and region per job to avoid cross-bin interaction issues
for indx in {3..6}; do
    for region in N S noDES SnoDES; do
        echo ">>> Processing indx=$indx region=$region"
        srun -n 4 python blinded_data_pip.py --indx $indx --regions $region
    done
done

'''
srun -n 4 python blinded_data_pip.py --regions NGC
srun -n 4 python blinded_data_pip.py --regions SGC
srun -n 4 python blinded_data_pip.py --regions GCcomb
'''