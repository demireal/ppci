import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures


def fit_obs_outcome_fn(df, regressors="X", target="T"):
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    y = np.array(df[target])

    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5)
    model.fit(X, y)

    return model


def fit_trial_outcome_fn(df, regressors=["X", "fa(X)"], target="T"):
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    y = np.array(df[target])

    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5)
    model.fit(X, y)

    return model


def fit_trial_bias_fn(df, regressors="X", target="Z"):
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    z = np.array(df[target])

    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5)
    model.fit(X, z)

    return model