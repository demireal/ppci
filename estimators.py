import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from utils import *


def bsl1_os_om(df_comp):
    """
    Baseline 1: Average the predictions of the predictive model f^a(x) (trained on observational data) in target sample.
    """
    df_tar = df_comp.query("S == 0")
    hat_mu = df_tar["fa(X)"].mean()

    return hat_mu


def bsl2_om(df_comp):
    """
    Baseline 2. Average the predictions of the outcome model g^a(X) in target sample (S=0), which is estimated from the trial sample (S=1).
    """
    df_tar = df_comp.query("S == 0")
    hat_mu = df_tar["ga(X)"].mean()

    return hat_mu


def bsl3_ipw(df_comp):
    """
    Baseline 3. Weighting adjustment in the trial.
    """
    num = sum(df_comp["Y"] * df_comp["w(XSA)"])
    denum = sum(df_comp["w(XSA)"])
    hat_mu = num / denum

    return hat_mu


def bsl4_dr(df_comp):
    """
    Baseline 4. DR estimator using trial data only.
    """
    hat_p = df_comp["S"].mean()
    n = len(df_comp)

    df_comp["res"] = df_comp["Y"] - df_comp["ga(X)"]
    
    df_comp["sum_i"] = df_comp.apply(lambda r: \
        (1 - r["S"]) * r["ga(X)"] + r["w(XSA)"] * r["res"], axis=1)
    
    hat_mu = sum(df_comp["sum_i"]) / (n * (1 - hat_p))

    return hat_mu


def nm1_abc(df_comp):
    """
    New model 1. Bias Model - Additive Bias Correction 
    """
    df_tar = df_comp.query("S == 0")

    mean_fax = df_tar["fa(X)"].mean()
    mean_bax = df_tar["ba(X)"].mean()
    hat_mu = mean_fax - mean_bax

    return hat_mu


def nm2_aom(df_comp):
    """
    New model 2. Augmented Outcome Model
    """
    df_tar = df_comp.query("S == 0")
    hat_mu = df_tar["ha(X)"].mean()
    
    return hat_mu


def nm3_dr_abc(df_comp):
    """
    New method 3. DR estimator for additive bias correction.
    """
    hat_p = df_comp["S"].mean()
    n = len(df_comp)

    df_comp["res"] = df_comp["Z"] - df_comp["ba(X)"]
    
    df_comp["sum_i"] = df_comp.apply(lambda r: \
        (1 - r["S"]) * r["ba(X)"] + r["w(XSA)"] * r["res"], axis=1)
    
    st = sum(df_comp["sum_i"]) / (n * (1 - hat_p))
    ft = bsl1_os_om(df_comp)
    hat_mu = ft - st

    return hat_mu


def nm4_dr_aom(df_comp):
    """
    New method 4. DR estimator for augmented outcome model.
    """
    hat_p = df_comp["S"].mean()
    n = len(df_comp)

    df_comp["res"] = df_comp["Y"] - df_comp["ha(X)"]
    
    df_comp["sum_i"] = df_comp.apply(lambda r: \
        (1 - r["S"]) * r["ha(X)"] + r["w(XSA)"] * r["res"], axis=1)
    
    hat_mu = sum(df_comp["sum_i"]) / (n * (1 - hat_p))

    return hat_mu

