import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from utils import *


def bsl1_avg_faX(df_comp):
    """
    Baseline 1.
    Average the predictions of the predictive model f^a(X) (trained on observational data) in target sample.
    """
    df_tar = df_comp.query("S == 0")
    hat_mu = df_tar["fa(X)"].mean()

    return hat_mu


def bsl2_avg_gaX(df_comp, gax_model="linear", hls=(128,32,8), activation="tanh", X_test=np.linspace(-1,1,51), poly_degree=10):
    """
    Baseline 2.
    Average the predictions of the outcome model estimated from the trial sample (S=1) in target sample (S=0).
    """
    g_a_X = fit_trial_outcome_fn(df_comp.copy(), regressors="X", target="Y", model=gax_model, hls=hls, activation=activation, poly_degree=poly_degree)
    df_tar = df_comp.query("S == 0")

    X_tar = np.array(df_tar['X']).reshape(-1,1)
    X_test = X_test.reshape(-1,1)

    if gax_model == "poly":
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_tar = poly.fit_transform(X_tar)
        X_test = poly.fit_transform(X_test)

    y_preds = g_a_X.predict(X_tar)
    y_test = g_a_X.predict(X_test)

    hat_mu = np.mean(y_preds)

    return hat_mu, y_test


def nm1_bm_abc(df_comp, bax_model="linear", hls=(128,32,8), activation="tanh", X_test=np.linspace(-1,1,51), poly_degree=10):
    """
    (Ours 1) New model 1. (Bias Model - Additive Bias Correction)
    Step 1. Average the predictions of the predictive model f^a(X) (trained on observational data) in target sample (Baseline 1 exactly).
    Step 2. Fit a "bias function" for the trial using the errors (Z) of the predictive model in the trial sample (S=1)
    Step 3. Average the bias function in the target sample (S=0) and subtract this from Step 1.
    """
    df_tar = df_comp.query("S == 0").copy()
    mean_fax = df_tar["fa(X)"].mean()

    b_a_X = fit_trial_bias_fn(df_comp.copy(), regressors="X", target="Z", model=bax_model, hls=hls, activation=activation, poly_degree=poly_degree)

    X_tar = np.array(df_tar['X']).reshape(-1,1)
    X_test = X_test.reshape(-1,1)

    if bax_model == "poly":
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_tar = poly.fit_transform(X_tar)
        X_test = poly.fit_transform(X_test)
        
    z_preds = b_a_X.predict(X_tar)
    z_test = b_a_X.predict(X_test)

    mean_z = np.mean(z_preds)
    hat_mu = mean_fax - mean_z

    return hat_mu, z_test


def nm2_om_pa(df_comp, hax_model="linear", hls=(128,32,8), activation="tanh", X_test=np.linspace(-1,1,51), f_a_X=None, poly_degree=10):
    """
    (Ours 2) New model 2. (Outcome Model - Prognostics Adjustment)
    Average the predictions of the augmented outcome model estimated from the trial sample (S=1) in target sample (S=0).
    """
    df_tar = df_comp.query("S == 0").copy()
    h_a_X = fit_trial_outcome_fn(df_comp.copy(), regressors=["X", "fa(X)"], target="Y", model=hax_model, hls=hls, activation=activation, poly_degree=poly_degree)
    
    f_a_X_test = f_a_X.predict(X_test.reshape(-1,1))

    X_tar_aug = np.array(df_tar[["X", "fa(X)"]]).reshape(-1,2)
    X_test_aug = np.array([X_test, f_a_X_test]).T

    if hax_model == "poly":
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_tar_aug = np.hstack((poly.fit_transform(X_tar_aug[:,0].reshape(-1,1)), X_tar_aug[:,1].reshape(-1,1)))
        X_test_aug = np.hstack((poly.fit_transform(X_test_aug[:,0].reshape(-1,1)), X_test_aug[:,1].reshape(-1,1)))

    y_preds = h_a_X.predict(X_tar_aug)
    y_test = h_a_X.predict(X_test_aug)

    hat_mu = np.mean(y_preds)    

    return hat_mu, y_test
