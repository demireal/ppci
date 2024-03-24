import sys, os, time
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from SyntheticDataModule import *
from estimators import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def sim_cases(case_idx, gp_params, params,
                all_covs, adj_covs, big_n_rct, n_tar, n_obs, n_MC, X_range, U_range, pasx,
                num_runs_per_case, fax_noise, seed, save_dir):

        SyntheticData = SyntheticDataModule(big_n_rct, n_tar, n_obs, n_MC, gp_params, all_covs, X_range, U_range, pasx, seed + 3)
        df_comp_big, df_obs = SyntheticData.get_df() 
        true_gax, true_psx = SyntheticData.plot_om(save_dir=f"{save_dir}/case_{case_idx}")
        mu_a_gt, _, _, _ = SyntheticData.get_true_mean()

        if not fax_noise:
            pars = {"degree": params["os-om"]["poly_degrees"][0], "Cs": params["os-om"]["Cs"], "cv_folds": params["os-om"]["cv_folds"], "penalty": params["os-om"]["penalty"]}
            f_a_X = regression_model(df_obs.query("A == 1"), adj_covs, "Y", "poly", pars)
            X_arr = np.array(df_comp_big[adj_covs]).reshape(-1, len(adj_covs))
            poly = PolynomialFeatures(degree=pars["degree"], include_bias=False)
            pcol = [f"Xp{i+1}" for i in range(pars["degree"])]
            df_comp_big[pcol] = poly.fit_transform(X_arr)
            df_comp_big["fa(X)"] = f_a_X.predict(np.array(df_comp_big[pcol]))
        else:
            f_a_X = "noise"
            df_comp_big["fa(X)"] = 5 * np.random.randn(len(df_comp_big))
        
        df_comp_big["Z"] = df_comp_big["fa(X)"] - df_comp_big["Y"]

        tar_idx = df_comp_big.query("S==0").index
        rct_idx_list = np.split(df_comp_big.query("S==1").index, num_runs_per_case)

        estimates = get_estimates(df_comp_big, tar_idx, rct_idx_list, f_a_X, true_gax, true_psx, case_idx, params["poly"], adj_covs, X_range, num_runs_per_case, save_dir)
        #plot_case_rmse(save_dir, case_idx, estimates, mu_a_gt)

        stat_bias_sq_est = np.mean(estimates - mu_a_gt, axis=0) ** 2
        stat_var_est = np.std(estimates, axis=0) ** 2
        mse = np.mean((estimates - mu_a_gt) ** 2, axis=0)
        rmse = np.sqrt(mse)        

        return stat_bias_sq_est, stat_var_est, rmse

def get_estimates(df_comp_big, tar_idx, rct_idx_list, f_a_X, true_gax, true_psx, case_idx, params, adj_covs, X_range, num_runs_per_case, save_dir):
    nem = 4
    num_estimates = len(params["poly_degrees"]) * nem
    estimates = np.zeros((num_runs_per_case, num_estimates))
    preds = {}

    for run_idx in range(num_runs_per_case):
        comp_idx = np.append(tar_idx, rct_idx_list[run_idx])
        df_comp = df_comp_big.loc[comp_idx, :].copy().reset_index(drop=True)
        
        for p_idx, pdeg in enumerate(params["poly_degrees"]):
            pars = {"degree": pdeg, "Cs": params["Cs"], "cv_folds": params["cv_folds"], "penalty": params["penalty"]}

            g_a_X = regression_model(df_comp.query("S == 1 & A == 1"), adj_covs, "Y", "poly", pars)
            h_a_X = regression_model(df_comp.query("S == 1 & A == 1"), adj_covs + ["fa(X)"], "Y", "poly", pars)
            b_a_X = regression_model(df_comp.query("S == 1 & A == 1"), adj_covs, "Z", "poly", pars)

            X_arr = np.array(df_comp[adj_covs]).reshape(-1, len(adj_covs))
            poly = PolynomialFeatures(degree=pdeg, include_bias=False)
            pcol = [f"Xp{i+1}" for i in range(pdeg)]
            df_comp[pcol] = poly.fit_transform(X_arr)
                                
            df_comp["ga(X)"] = g_a_X.predict(df_comp[pcol])
            df_comp["ba(X)"] = b_a_X.predict(df_comp[pcol])
            df_comp["ha(X)"] = h_a_X.predict(df_comp[pcol + ["fa(X)"]])

            estimates[run_idx, nem * p_idx + 0] = bsl1_os_om(df_comp)
            estimates[run_idx, nem * p_idx + 1] = bsl2_om(df_comp)
            estimates[run_idx, nem * p_idx + 2] = nm1_abc(df_comp)
            estimates[run_idx, nem * p_idx + 3] = nm2_aom(df_comp)

            preds[f"gax_pd_{pdeg}"], preds[f"bax_pd_{pdeg}"], preds[f"hax_pd_{pdeg}"] =\
                  get_om_fits(g_a_X, b_a_X, h_a_X, f_a_X, adj_covs, "poly", pdeg, params["poly_degrees"][-1], X_range)

    for pdeg in params["poly_degrees"]:
        plot_case_oms(f"{save_dir}/case_{case_idx}", X_range, df_comp, f_a_X, true_gax, true_psx, preds, pdeg, params["poly_degrees"][-1])

    return estimates


def plot_case_oms(save_dir, X_range, df_comp, f_a_X, true_gax, true_psx, preds, pdeg, os_pdeg):

    matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': 16})

    os.makedirs(save_dir, exist_ok=True)

    X_test = X_range.reshape(-1, 1)

    poly = PolynomialFeatures(degree=os_pdeg, include_bias=False)
    fX_test = poly.fit_transform(X_test)

    if f_a_X == None:
        fax_preds = 5 * np.random.randn(len(X_test))
    else:
        fax_preds = f_a_X.predict(fX_test)
    true_bax = fax_preds - true_gax

    cp = ["red", "#34b6c6", "mediumpurple","#79ad41", "crimson", "navy", "black", "goldenrod"]
    _, (ax1,ax2,ax3) = plt.subplots(3, 1,gridspec_kw={'height_ratios': [0.5, 1, 1]})

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.plot(X_test.reshape(-1), true_psx, color=cp[4], label='P(S=1 | X)')
    ax1.plot(X_test.reshape(-1), 1 - true_psx, color=cp[5], label='P(S=0 | X)')
    ax1.fill_between(X_test.reshape(-1), true_psx, np.zeros(26), color=cp[4], alpha=0.2, label='P(S=1 | X)')
    ax1.fill_between(X_test.reshape(-1), 1 - true_psx, np.zeros(26), color=cp[5], alpha=0.2, label='P(S=0 | X)')
    ax1.tick_params(axis='x', colors='dimgray')
    ax1.tick_params(axis='y', colors='dimgray')

    ax2.set_xticks([])
    ax2.plot(X_test, true_gax, label=r'$g_1 (X)$', color=cp[6], linewidth=2)
    ax2.plot(X_test, preds[f"gax_pd_{pdeg}"], label=r'$\hat{g}_1 (X)$', ls='-.', color=cp[1], linewidth=2)
    ax2.plot(X_test, preds[f"hax_pd_{pdeg}"], label=r'$\hat{h}_1 (\tilde{X})$', ls='--', color=cp[2], linewidth=2)
    ax2.plot(X_test, fax_preds, label=r'$f_1 (X)$', color=cp[0], ls=':', linewidth=2)
    ax2.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Y"], s=3, color=cp[7], alpha=1, label=r"Trial patients ($Y$)")
    ax2.tick_params(axis='x', colors='dimgray')
    ax2.tick_params(axis='y', colors='dimgray')

    ax3.plot(X_test, true_bax, label=r'$b_1 (X) = f_1 (X) - g_1 (X)$', color='dimgray', linewidth=2)
    ax3.plot(X_test, preds[f"bax_pd_{pdeg}"], label=r'$\hat{b}_1 (X)$', color=cp[3], ls='--', linewidth=2)
    ax3.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Z"], s=3, color=cp[7], alpha=1, label=r"Trial patients ($Z$)")
    ax3.tick_params(axis='x', colors='dimgray')
    ax3.tick_params(axis='y', colors='dimgray')
    ax3.set_xlabel(r"$X$")

    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(save_dir, f"example_img_pd_{pdeg}.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"example_img_pd_{pdeg}.png"), bbox_inches='tight')
    plt.close()


def plot_case_rmse(save_dir, case_idx, estimates, mu_a_gt):

    rmse_full = np.sqrt((estimates - mu_a_gt) ** 2)
    rmse = rmse_full[:,1:]

    mean = np.mean(rmse, axis=0)
    std = np.std(rmse, axis=0) / len(rmse)

    poly_degs = [1,4,7,10]
    methods = ["gax","bax","hax","fax"]
    fb_alpha = 0.25
    lw = 2
    cp = ["salmon", "#34b6c6", "mediumpurple","#79ad41"]
    cp_ind = {"fax": 0, "gax": 1, "bax": 3, "hax": 2}
    labels = {"fax": r"$\hat{\mu}^a_{OBS-OM}$", "gax": r"$\hat{\mu}^a_{OM}$", "bax": r"$\hat{\mu}^a_{ABC}$", "hax": r"$\hat{\mu}^a_{AOM}}$"}
    markers = {"fax": " ", "gax": " ", "bax": " ", "hax": " "}
    line_styles = {"fax": "-", "gax": "-", "bax": "-", "hax": "-"}
    fill_styles = {"fax": "full", "gax": "none", "bax": "none", "hax": "full"}

    plt.figure()
    sns.set_style("whitegrid")

    for mind,met in enumerate(methods):
        if met == 'fax':
            mean = np.mean(rmse_full, axis=0)
            std = np.std(rmse_full, axis=0) / len(rmse)
            mean_rmse = np.array([mean[0], mean[0], mean[0], mean[0]])
            std_rmse = np.array([std[0], std[0], std[0], std[0]])
            lb = mean_rmse - 2 * std_rmse
            ub = mean_rmse + 2 * std_rmse
        else:
            mean_rmse = mean[mind::3]
            std_rmse = std[mind::3]
            lb = mean_rmse - 2 * std_rmse
            ub = mean_rmse + 2 * std_rmse
        plt.plot(poly_degs, mean_rmse, color=cp[cp_ind[met]], linewidth=lw, linestyle=line_styles[met], marker=markers[met], fillstyle=fill_styles[met])
        plt.fill_between(poly_degs, lb, ub, color=cp[cp_ind[met]], alpha=fb_alpha)

    plt.xticks(poly_degs)
    plt.xlabel("Degree of polynomial fit")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(f"{save_dir}/case_{case_idx}/fig.png"), bbox_inches="tight")
    plt.savefig(os.path.join(f"{save_dir}/case_{case_idx}/fig.svg"), bbox_inches="tight")
    plt.close()


def save_setting_stats(results, save_dir, poly_degrees):

    n = len(results)
    base_methods = ["fax", "om-gax", "om-bax", "om-hax"]
    methods = [*np.concatenate([[bm + "-PD-" + str(pd) for bm in base_methods] for pd in poly_degrees])]

    bias_sq_arr = np.vstack([tpl[0] for tpl in results])
    var_arr = np.vstack([tpl[1] for tpl in results])
    rmse_arr = np.vstack([tpl[2] for tpl in results])
    
    avg_bias_sq, std_bias_sq = np.mean(bias_sq_arr, axis=0), np.std(bias_sq_arr, axis=0) / n
    avg_var, std_var = np.mean(var_arr, axis=0), np.std(var_arr, axis=0) / n
    avg_rmse, std_rmse = np.mean(rmse_arr, axis=0), np.std(rmse_arr, axis=0) / n
    
    data = np.vstack((avg_rmse, avg_bias_sq, avg_var)).T
    data_wstd = np.vstack((avg_rmse, std_rmse, avg_bias_sq, std_bias_sq, avg_var, std_var)).T

    df_res = pd.DataFrame(data, index=methods, columns=["RMSE", "Squared-Bias", "Variance"])
    df_res.to_csv(os.path.join(save_dir, f"res.csv"), float_format='%.5f')

    df_res_wstd = pd.DataFrame(data_wstd, index=methods, columns=["RMSE", "Std.Dev.", "Squared-Bias", "Std.Dev.", "Variance", "Std.Dev."])
    df_res_wstd.to_csv(os.path.join(save_dir, f"res_wstd.csv"), float_format='%.5f')
