import numpy as np
import pandas as pd
import random, os
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import expit
from utils import * 


class CombinedDataModule:
    def __init__(self, 
                seed=42,
                d=2, 
                n_rct=200,
                n_tar=2000,
                n_obs=10000,
                n_MC=100000,
                X_range=np.linspace(-1,1,51),
                U_range=np.linspace(-1,1,51),
                om_A0=None,
                om_A1=None,
                w_sel=None,
                w_trt=None,
                ):

        self.seed = seed
        self.d = d  # covariates dimensionality (integer)
        self.n_rct = n_rct
        self.n_tar = n_tar
        self.n_obs = n_obs
        self.n = 10 * (n_rct + n_tar)  # auxiliary variable used in data generating
        self.n_MC = n_MC  # monte-carlo sample size to calculate "true mean" in the target population
        self.X = X_range
        self.U = U_range

        self.covs= ["X", "U"]
        self.XX, self.UU = np.meshgrid(self.X, self.U)
        self.XU_flat = np.c_[self.XX.ravel(), self.UU.ravel()]

        self.om_A0 = om_A0  # GP for the outcome model under treatment A=0
        self.om_A1 = om_A1  # GP for the outcome model under treatment A=1
        self.w_sel = w_sel  # GP for the selection score model P(S=1 | X)
        self.w_trt = w_trt  # GP for the propensity score in OBS study P(A=1 | X, S=2)

        np.random.seed(self.seed)
        self.df  = self._generate_data()
        self.df_obs = self._generate_data_obs()


    def _generate_data(self):  
        complete = False

        while not complete:
            try:
                df = pd.DataFrame(index=np.arange(self.n))
                df[self.covs] = 2 * np.random.rand(self.n, self.d) - 1  # Uniform[-1,1]

                df["P(S=1|X)"] = df.apply(lambda row: np.clip(expit(self.w_sel(row["X"], row["U"])[0]), 0.1, 0.9), axis=1)
                df["S"] = np.array(df["P(S=1|X)"] > np.random.uniform(size=self.n), dtype=int)  # selection into trial via sampling from Bernoulli(P(S=1|X))

                rct_idx = random.sample(df.index[df["S"] == 1].tolist(), self.n_rct)
                tar_idx = random.sample(df.index[df["S"] == 0].tolist(), self.n_tar)

                complete=True
            except:
                print("generating data...")
                self.n = 2 * self.n

        df = df.loc[rct_idx + tar_idx, :].copy().reset_index(drop=True)

        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)

        df.loc[df.S == 1, "A"] = np.array(0.5 > np.random.uniform(size=self.n_rct), dtype=int)  # random sampling treatment with probability 1/2 for A=1

        df.loc[df.S == 1, "Y"] = df.loc[df.S == 1, "Y1"] * df.loc[df.S == 1, "A"] +\
                                df.loc[df.S == 1, "Y0"] * (1 - df.loc[df.S == 1, "A"])

        df = df[["S", "A"] + self.covs + ["Y"]].sort_values(by="S").reset_index(drop=True).copy()

        return df
    

    def _generate_data_obs(self):  

        df = pd.DataFrame(index=np.arange(self.n_obs))
        df[self.covs] = 2 * np.random.rand(self.n_obs, self.d) - 1

        df["P(A=1|X)"] = df.apply(lambda row: np.clip(expit(self.w_trt(row["X"], row["U"])[0]), 0.1, 0.9), axis=1)
        df["A"] = np.array(df["P(A=1|X)"] > np.random.uniform(size=self.n_obs), dtype=int) 

        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)

        df["Y"] = df["Y1"] * df["A"] + df["Y0"] * (1 - df["A"])

        df = df[self.covs + ["A", "Y"]].reset_index(drop=True).copy()

        return df


    def get_df(self):
        return self.df.copy(), self.df_obs.copy()
    

    def get_true_mean(self, print_res=False):  
        np.random.seed(self.seed + 1)
        df = pd.DataFrame(index=np.arange(self.n_MC))
        df[self.covs] = 2 * np.random.rand(self.n_MC, self.d) - 1  # Uniform[-1,1]

        df["P(S=1|X)"] = df.apply(lambda row: expit(self.w_sel(row["X"], row["U"])[0]), axis=1)
        df["S"] = np.array(df["P(S=1|X)"] > np.random.uniform(size=self.n_MC), dtype=int)  # selection into trial via sampling from Bernoulli(P(S=1|X))

        df['Y0'] = df.apply(lambda row: self.om_A0(row["X"], row["U"])[0], axis=1)
        df['Y1'] = df.apply(lambda row: self.om_A1(row["X"], row["U"])[0], axis=1)

        n_MC_tar = len(df.loc[df.S == 0])
        n_MC_rct = len(df.loc[df.S == 1])

        mean_S0Y1, std_S0Y1 = df.loc[df.S == 0, "Y1"].mean(), df.loc[df.S == 0, "Y1"].std() / n_MC_tar
        mean_S1Y1, std_S1Y1 = df.loc[df.S == 1, "Y1"].mean(), df.loc[df.S == 1, "Y1"].std() / n_MC_rct

        if print_res:
            print(f"MC sample sizes: target = {n_MC_tar} and rct = {n_MC_rct}")
            print(f"Target pop. mean: {mean_S0Y1:.3f} +- {std_S0Y1:.3f}")
            print(f"Trial pop. mean: {mean_S1Y1:.3f} +- {std_S1Y1:.3f}")

        return mean_S0Y1, std_S0Y1, mean_S1Y1, std_S1Y1
    

    def get_some_stats(self):  
        A1_mean = self.df_obs["A"].mean()
        U_mean = self.df_obs[self.df_obs.A == 1]["U"].mean()
        X_mean = self.df_obs[self.df_obs.A == 1]["X"].mean()

        print(f"A mean (OBS study): {A1_mean:.2f}")
        print(f"U mean (OBS study): {U_mean:.2f}")
        print(f"X mean (OBS study): {X_mean:.2f}")
    
    
    def plot_om(self, save=False, save_dir=None):

        matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
        matplotlib.rcParams['ps.fonttype'] = 42

        Yp = self.om_A1(self.X, self.U)
        psx = np.clip(expit(self.w_sel(self.X, self.U)), 0.1, 0.9)
        pax = np.clip(expit(self.w_trt(self.X, self.U)), 0.1, 0.9)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))

        contour = ax1.contourf(self.XX, self.UU, Yp, cmap='RdBu')
        ax1.set_xlabel('X')
        ax1.set_ylabel('U')
        plt.colorbar(contour, ax=ax1)

        ax2 = fig.add_subplot(322, projection='3d')
        ax2.plot_surface(self.XX, self.UU, Yp, cmap='RdBu')
        ax2.set_xlabel('X')
        ax2.set_ylabel('U')
        ax2.set_zlabel(f'E[Y1 | X, U]')

        avg_val_x = np.mean(Yp, axis=0)
        ax3.plot(self.X, avg_val_x)
        ax3.set_xlabel('X')
        ax3.set_ylabel(f'E[Y1 | X]')

        avg_val_u = np.mean(Yp, axis=1)
        ax4.plot(self.U, avg_val_u)
        ax4.set_xlabel('U')
        ax4.set_ylabel(f'E[Y1 | U]')

        psx_avg = np.mean(psx, axis=0)
        ax5.plot(self.X, psx_avg)
        ax5.set_xlabel('X')
        ax5.set_ylabel('P(S=1 | X)')

        ax6.plot(self.U, np.mean(pax, axis=1))
        ax6.set_xlabel('U')
        ax6.set_ylabel('P(A=1 | U, S=2 (OBS))')

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, "true_dgp.jpg")
        plt.savefig(img_path, bbox_inches='tight')

        return avg_val_x, psx_avg




    
