#!/bin/bash

# !!! Process one catalog bin and region per job to avoid cross-bin interaction issues
for indx in {0..6}; do
    for region in NGC SGC GCcomb; do
        echo ">>> Processing indx=$indx region=$region"
        srun -n 4 python blinded_data_pip.py --version dr2-v2 --subver zcmb --indx $indx --regions $region --todo blinded_mesh2_spectrum
    done
done