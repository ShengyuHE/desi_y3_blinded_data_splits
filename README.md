# Data splits tests

This repository provides all the analysis done for **DESI DR2 blinded data splits tests**.
Measurements and results are saved in desi path: 

''
/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/blinded_data/dr2-v2/data_splits
''

To activate the envrioment in NERSC, suggest:
```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
```
Personal enviroment: 
```
source /global/homes/s/shengyu/env.sh 2pt_env
```

## Power spectrum estimator

We use `jaxpower` (https://github.com/adematti/jax-power.git) to estimate the power spectrum and compute the window function and analytical Gaussian covariance matrix. The original script of `blinded_data_pip.py` is in （https://github.com/cosmodesi/cai-mock-benchmark/blob/a5c32261a5a74fa248c2dedaa503871d57c87389/dr2/data_pip.py）
