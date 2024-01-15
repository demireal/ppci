from CombinedDataModule import *
from estimators import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def sim_one_case(case_idx, seed, save_dir,  om_A0_par, om_A1_par, w_sel_par, w_trt_par, 
                 d, big_n_rct, n_tar, n_obs, n_MC, num_est, num_runs_per_case, X_range, U_range, methods):

        estimates = np.zeros((num_runs_per_case, num_est))
        preds = {}

        om_A0 = sample_outcome_model_gp(om_A0_par, X_range, U_range, seed)  # GP - for the outcome model under treatment A=1
        om_A1 = sample_outcome_model_gp(om_A1_par, X_range, U_range, seed+1)  # GP - for the outcome model under treatment A=1
        w_sel = sample_outcome_model_gp(w_sel_par, X_range, U_range, seed+2)  # GP - for the selection score model P(S=1 | X)
        w_trt = sample_outcome_model_gp(w_trt_par, X_range, U_range, seed+3)  # GP - for the propensity score in OBS study P(A=1 | X, S=2)

        CombinedData = CombinedDataModule(seed + 4, d, big_n_rct, n_tar, n_obs, n_MC, X_range, U_range, om_A0, om_A1, w_sel, w_trt)
        df_comp_big, df_obs = CombinedData.get_df() 
        true_gax, true_psx = CombinedData.plot_om(save_dir=f"{save_dir}/case_{case_idx}")
        mu_a_gt, _, _, _ = CombinedData.get_true_mean(print_res=False)

        f_a_X = fit_obs_outcome_fn(df_obs, regressors="X", target="Y", model="NN", hls=(256,64,16), activation="tanh")
        df_comp_big['fa(X)'] = f_a_X.predict(np.array(df_comp_big['X']).reshape(-1,1))
        df_comp_big['Z'] = df_comp_big['fa(X)'] - df_comp_big['Y']

        tar_idx = df_comp_big.query("S==0").index
        rct_idx = np.split(df_comp_big.query("S==1").index, num_runs_per_case)

        for run_idx in range(num_runs_per_case):
            comp_idx = np.append(tar_idx, rct_idx[run_idx])
            df_comp = df_comp_big.loc[comp_idx, :].copy().reset_index(drop=True)

            estimates[run_idx, 0] = bsl1_avg_faX(df_comp.copy())
            estimates[run_idx, 1], preds["gax_preds_lin"] = bsl2_avg_gaX(df_comp.copy(), gax_model="linear", X_test=X_range)
            estimates[run_idx, 2], preds["bax_preds_lin"] = nm1_bm_abc(df_comp.copy(), bax_model="linear", X_test=X_range)
            estimates[run_idx, 3], preds["hax_preds_lin"] = nm2_om_pa(df_comp.copy(), hax_model="linear", X_test=X_range, f_a_X=f_a_X)
            estimates[run_idx, 4], preds["gax_preds_poly"] = bsl2_avg_gaX(df_comp.copy(), gax_model="poly", X_test=X_range, poly_degree=10)
            estimates[run_idx, 5], preds["bax_preds_poly"] = nm1_bm_abc(df_comp.copy(), bax_model="poly", X_test=X_range, poly_degree=10)
            estimates[run_idx, 6], preds["hax_preds_poly"] = nm2_om_pa(df_comp.copy(), hax_model="poly", X_test=X_range, f_a_X=f_a_X, poly_degree=10)

        stat_bias_sq_est = np.mean(estimates - mu_a_gt, axis=0) ** 2
        stat_var_est = np.std(estimates, axis=0) ** 2
        mse = np.mean((estimates - mu_a_gt) ** 2, axis=0)
        rmse = np.sqrt(mse)

        np.savetxt(os.path.join(f"{save_dir}/case_{case_idx}", f"example_bias.txt"), estimates[-1,:] - mu_a_gt, fmt='%1.4f')
        plot_case(f"{save_dir}/case_{case_idx}", X_range, df_comp, df_obs, f_a_X, true_gax, true_psx, preds, "lin")
        plot_case(f"{save_dir}/case_{case_idx}", X_range, df_comp, df_obs, f_a_X, true_gax, true_psx, preds, "poly")

        return stat_bias_sq_est, stat_var_est, rmse


def plot_case(save_dir, X_range, df_comp, df_obs, f_a_X, true_gax, true_psx, preds, pred_type):

    matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': 16})

    os.makedirs(save_dir, exist_ok=True)

    X_test = X_range.reshape(-1, 1)
    fax_preds = f_a_X.predict(X_test)
    true_bax = fax_preds - true_gax

    cp = sns.color_palette("tab10")
    _, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(8, 12),gridspec_kw={'height_ratios': [0.33, 1, 1, 1]})

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.fill_between(X_test.reshape(-1), true_psx, np.zeros(51), color=cp[2], alpha=0.25, label='P(S=1 | X)')
    ax1.fill_between(X_test.reshape(-1), 1 - true_psx, np.zeros(51), color=cp[6], alpha=0.25, label='P(S=0 | X)')
    ax1.legend(fontsize="12")

    ax2.set_xticks([])
    ax2.plot(X_test, true_gax, label=r'$g_{s=1}^a(X)$', color=cp[7], linewidth=2)
    ax2.plot(X_test, preds[f"gax_preds_{pred_type}"], label=r'$\hat{g}_{s=1}^a(X)$', ls='--', color=cp[2], linewidth=2)
    ax2.plot(X_test, preds[f"hax_preds_{pred_type}"], label=r'$\hat{h}_{s=1}^a(\tilde{X})$', ls='--', color=cp[0], linewidth=2)
    ax2.plot(X_test, fax_preds, label=r'$f^a(X)$', color=cp[3], ls='-.', linewidth=2)
    ax2.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Y"], s=3, color=cp[-2], alpha=1, label=r"Trial patients ($Y$)")
    ax2.legend(fontsize="12")

    ax3.set_xticks([])
    ax3.plot(X_test, true_bax, label=r'$b_{s=1}^a(X) = f^a(X) - g_{s=1}^a(X)$', color=cp[-1], linewidth=2)
    ax3.plot(X_test, preds[f"bax_preds_{pred_type}"], label=r'$\hat{b}_{s=1}^a(X)$', color=cp[6], ls='--', linewidth=2)
    ax3.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Z"], s=3, color=cp[-2], alpha=1, label=r"Trial patients ($Z$)")
    ax3.legend(fontsize="12")

    ax4.axhline(0, color='gray')
    ax4.plot(X_test, preds[f"gax_preds_{pred_type}"].reshape(-1) - true_gax, label=r'$\hat{g}_{s=1}^a(X) - g_{s=1}^a(X)$', ls='dotted', color=cp[2], linewidth=3)
    ax4.plot(X_test, preds[f"hax_preds_{pred_type}"].reshape(-1) - true_gax, label=r'$\hat{h}_{s=1}^a(\tilde{X}) - h_{s=1}^a(\tilde{X})$', ls='dotted', color=cp[0], linewidth=3)
    ax4.plot(X_test, preds[f"bax_preds_{pred_type}"].reshape(-1) - true_bax, label=r'$\hat{b}_{s=1}^a(X) - b_{s=1}^a(X)$', ls='dotted', color=cp[6], linewidth=3)
    ax4.legend(fontsize="12")
    ax4.set_xlabel(r"$X$")

    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(save_dir, f"example_img_{pred_type}.png"), bbox_inches='tight')

def save_setting_stats(results, save_dir, methods):

    n = len(results)

    rmse_arr = np.vstack([tpl[2] for tpl in results])
    bias_sq_arr = np.vstack([tpl[0] for tpl in results])
    var_arr = np.vstack([tpl[1] for tpl in results])
    
    avg_rmse, std_rmse = np.mean(rmse_arr, axis=0), np.std(rmse_arr, axis=0) / n
    avg_bias_sq, std_bias_sq = np.mean(bias_sq_arr, axis=0), np.std(bias_sq_arr, axis=0) / n
    avg_var, std_var = np.mean(var_arr, axis=0), np.std(var_arr, axis=0) / n
    
    data = np.vstack((avg_rmse, avg_bias_sq, avg_var)).T
    data_wstd = np.vstack((avg_rmse, std_rmse, avg_bias_sq, std_bias_sq, avg_var, std_var)).T

    df_res = pd.DataFrame(data, index=methods, columns=["RMSE", "Squared-Bias", "Variance"])
    df_res.to_csv(os.path.join(save_dir, f"res.csv"), float_format='%.5f')

    df_res_wstd = pd.DataFrame(data_wstd, index=methods, columns=["RMSE", "Std.Dev.", "Squared-Bias", "Std.Dev.", "Variance", "Std.Dev."])
    df_res_wstd.to_csv(os.path.join(save_dir, f"res_wstd.csv"), float_format='%.5f')

