from CombinedDataModule import *
from estimators import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def sim_one_case(case_idx, random_seed, save_dir,  om_A0_par, om_A1_par, w_sel_par, w_trt_par, 
                 d, big_n_rct, n_tar, n_obs, n_MC, num_est, num_runs_per_case, X_range, U_range):

        estimates = np.zeros((num_runs_per_case, num_est))
        preds = {}

        om_A0 = sample_outcome_model_gp(om_A0_par, X_range, U_range)  # GP - for the outcome model under treatment A=1
        om_A1 = sample_outcome_model_gp(om_A1_par, X_range, U_range)  # GP - for the outcome model under treatment A=1
        w_sel = sample_outcome_model_gp(w_sel_par, X_range, U_range)  # GP - for the selection score model P(S=1 | X)
        w_trt = sample_outcome_model_gp(w_trt_par, X_range, U_range)  # GP - for the propensity score in OBS study P(A=1 | X, S=2)

        CombinedData = CombinedDataModule(random_seed, d, big_n_rct, n_tar, n_obs, n_MC, X_range, U_range, om_A0, om_A1, w_sel, w_trt)
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

        stat_bias_sq_est = np.mean((mu_a_gt - estimates), axis=0) ** 2
        stat_var_est = np.std(estimates, axis=0) ** 2
        direct_mse = np.mean((mu_a_gt - estimates) ** 2, axis=0)

        plot_case(f"{save_dir}/case_{case_idx}", X_range, df_comp, df_obs, f_a_X, true_gax, true_psx, preds)

        return stat_bias_sq_est, stat_var_est, direct_mse


def plot_case(save_dir, X_range, df_comp, df_obs, f_a_X, true_gax, true_psx, preds):

    matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
    matplotlib.rcParams['ps.fonttype'] = 42

    os.makedirs(save_dir, exist_ok=True)

    X_test = X_range.reshape(-1, 1)
    fax_preds = f_a_X.predict(X_test)
    bias = fax_preds - true_gax

    cp = sns.color_palette("tab10")
    dark_cp = sns.color_palette("dark")

    plt.figure()
    plt.plot(X_test, true_gax, label=r'$g_{s=1}^a(X)$', color=cp[0])
    plt.plot(X_test, fax_preds, label=r'$f^a(X)$', color=cp[2], ls='--')
    plt.scatter(df_obs.query("A==1")["X"],df_obs.query("A==1")["Y"], s=1, color=cp[7], alpha=0.15, label='Observational patients (S=2)')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.savefig(os.path.join(save_dir, "img_1.jpg"), bbox_inches='tight')

    plt.figure()
    plt.plot(X_test, true_gax, label=r'$g_{s=1}^a(X)$', color=cp[0])
    plt.plot(X_test, fax_preds, label=r'$f^a(X)$', color=cp[2], ls='--')
    plt.plot(X_test, preds["gax_preds_lin"], label=r'$\hat{g}_{s=1}^a(X)$', ls='--', color=cp[1])
    plt.plot(X_test, preds["hax_preds_lin"], label=r'$\hat{h}_{s=1}^a(\tilde{X})$', ls='--', color=dark_cp[3])
    #plt.plot(X_test, preds["gax_preds_nn"], label=r'$\hat{g}_{s=1}^a(X)$ (NN)', ls='--', color=cp[4])
    #plt.plot(X_test, preds["hax_preds_nn"], label=r'$\hat{h}_{s=1}^a(\tilde{X})$ (NN)', ls='--', color=dark_cp[8])
    plt.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Y"], s=1, color=cp[7], alpha=1, label="Trial patients")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.savefig(os.path.join(save_dir, "img_2.jpg"), bbox_inches='tight')

    plt.figure()
    plt.plot(X_test, bias, label=r'$b_{s=1}^a(X)$', color=cp[6])
    plt.plot(X_test, preds["bax_preds_lin"], label=r'$\hat{b}_{s=1}^a(X)$', color=cp[5], ls='-.')
    #plt.plot(X_test,preds["bax_preds_nn"], label=r'$\hat{b}_{s=1}^a(X)$ (NN)', color=cp[8], ls='-.')
    plt.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Z"], s=1, color=cp[7], alpha=1, label="Trial patients")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel("Z")
    plt.savefig(os.path.join(save_dir, "img_3.jpg"), bbox_inches='tight')

    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 3]})

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.fill_between(X_test.reshape(-1), true_psx, np.zeros(51), color=cp[2], alpha=0.25, label='P(S=1 | X)')
    ax1.fill_between(X_test.reshape(-1), 1 - true_psx, np.zeros(51), color=cp[4], alpha=0.25, label='P(S=0 | X)')

    ax1.legend()

    ax2.plot(X_test, true_gax, label=r'$g_{s=1}^a(X)$', color=cp[0])
    ax2.axhline(0, color='black')
    ax2.plot(X_test, preds["gax_preds_lin"], label=r'$\hat{g}_{s=1}^a(X)$', ls='--', color=cp[1])
    ax2.plot(X_test, preds["gax_preds_lin"].reshape(-1) - true_gax, label=r'$\hat{g}_{s=1}^a(X) - g_{s=1}^a(X)$', ls='--', color=cp[3])
    ax2.scatter(df_comp.query("S==1 & A==1")["X"],df_comp.query("S==1 & A==1")["Y"], s=1, color=cp[7], alpha=1, label="Trial patients")

    plt.subplots_adjust(hspace=-0.02)  # Adjust this value as needed

    plt.legend()
    plt.xlabel('X')
    plt.savefig(os.path.join(save_dir, "img_4.jpg"), bbox_inches='tight')


def get_setting_stats(results):

    bias_arr = np.vstack([tpl[0] for tpl in results])
    var_arr = np.vstack([tpl[1] for tpl in results])
    mse_arr = np.vstack([tpl[2] for tpl in results])

    avg_bias, std_bias = np.mean(bias_arr, axis=0), np.std(bias_arr, axis=0)
    avg_var, std_var = np.mean(var_arr, axis=0), np.std(var_arr, axis=0)
    avg_mse, std_mse = np.mean(mse_arr, axis=0), np.std(mse_arr, axis=0)

    return avg_bias, std_bias, avg_var, std_var, avg_mse, std_mse

