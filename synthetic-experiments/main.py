import numpy as np
import os, json, itertools
from joblib import Parallel, delayed

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from run_utils import *
from estimators import *

'''
    many parts break when > 2 dim. covariates are used:
        data-generating process and plotting of the real functions therein
        sampling from a 2 dim Gaussian process
    will try to generalize to higher dims.
'''

d = 2                           # 2-dim. covariate vector
n_MC = 50000                    # num. Monte Carlo samples to calculate ground truth mean in the target population
all_covs = ["X", "U"]           # all covariates used in the data generating process
adj_covs = ["X"]                # covariates used in learning the nuisance functions such as the outcome models and weights.
X_range = np.linspace(-1,1,51)  # range of the first covariate
U_range = np.linspace(-1,1,51)  # range of the first covariate

n_rct_list = [200, 1000]
n_tar = 2000                     # target sample size
n_obs = 50000                     # observational sample size
num_case_per_setting = 30         # num. different ground-truth cases to simulate for EACH GP param. setting
num_runs_per_case = 40           # num. runs for EACH ground-truth case. a new trial sample is drawn every time.

# n_rct_list = [200]
# n_tar = 2000                   # target sample size
# n_obs = 5000                   # observational sample size
# num_case_per_setting = 2       # num. different ground-truth cases to simulate for EACH GP param. setting
# num_runs_per_case = 10         # num. runs for EACH ground-truth case. a new trial sample is drawn every time.

om_A0_par_list = [{'ls':[1,1], 'alpha':[1,1], "kernel": "rbf"}]  # unused rn. 

om_A1_par_list = [{'ls':[0.5,0.5], 'alpha':[5,5], "kernel": "rbf"},        # GP parameters for FOM_1
                  {'ls':[0.2,0.5], 'alpha':[5,5], "kernel": "rbf"}]

w_sel_par_list = [{'ls':[1,1e6], 'alpha':[10,0], "kernel": "rbf"}]         # Nested trial participation P(S=1 | X,U) GP parameters

w_trt_par_list = [{'ls':[1e6,1e6], 'alpha':[0,0], "kernel": "rbf"},        # OS treatment assignment P(A=1 | S=2, X,U) GP parameters
                  {'ls':[1e6,0.5], 'alpha':[0,0], "kernel": "rbf"},
                  {'ls':[1e6,0.5], 'alpha':[0,10], "kernel": "rbf"}]

# om_A1_par_list = [{'ls':[0.2,0.5], 'alpha':[5,5], "kernel": "rbf"}]
# w_sel_par_list = [{'ls':[1,1e6], 'alpha':[10,0], "kernel": "rbf"}]         # Nested trial participation P(S=1 | X,U) GP parameters
# w_trt_par_list = [{'ls':[1e6,0.5], 'alpha':[0,10], "kernel": "rbf"}]

fax_noise = False
poly_degrees = [1,4,7,10]
pasx = {"lb":0.1, "ub":0.9, "trial":0.5}  # params for data-generating. P(S=1|X) is clipped by lb & ub and P(A=1|S=1) = pasx['trial']
params = {"poly": {"poly_degrees": poly_degrees, "Cs": [1e-3, 1e-2, 1e-1, 1], "penalty": "l2", "cv_folds": 5},
          "os-om": {"act": "tanh", "hls": (256,64,16), "early_stp": True, "val_frac": 0.2}}

for n_rct in n_rct_list:
    big_n_rct = num_runs_per_case * n_rct     # sample a big trial once and use its partitions in each run 
    np.random.seed(42)
    case_seeds = np.random.randint(1e6, size=num_case_per_setting)

    for ns, (om_A1_par, om_A0_par, w_sel_par, w_trt_par) in \
        enumerate(itertools.product(om_A1_par_list, om_A0_par_list, w_sel_par_list,  w_trt_par_list)):

        gp_params = {"om_A0_par": om_A0_par, "om_A1_par": om_A1_par, "w_sel_par": w_sel_par, "w_trt_par": w_trt_par}

        save_dir = f"./new_res_all/mpx_nrct_{n_rct}/gp_draw_{ns}"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"settings_{ns}.json"), 'w') as json_file:
            json_file.write(json.dumps(gp_params, indent=4))

        results = \
        Parallel(n_jobs=36)(
                    delayed(sim_cases) 
                    (
                    case_idx, gp_params, params,\
                    all_covs, adj_covs, big_n_rct, n_tar, n_obs, n_MC, X_range, U_range, pasx, \
                    num_runs_per_case, fax_noise, random_seed, save_dir \
                    ) 
                    for case_idx, random_seed in enumerate(case_seeds)
                )

        # sim_cases(1, gp_params, params,\
        #              all_covs, adj_covs, big_n_rct, n_tar, n_obs, n_MC, X_range, U_range, pasx, \
        #              num_runs_per_case, fax_noise, 42, save_dir) 
        
        save_setting_stats(results, save_dir, poly_degrees)