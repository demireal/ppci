def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from scipy import interpolate
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures


def rbf_linear_kernel(X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=5):  # works with 2D covariates only
    distances = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / length_scales, axis=2)
    rbf_term = var * np.exp(-0.5 * distances**2)
    linear_term = np.dot(np.dot(X1, np.diag(alpha)), X2.T)
    return rbf_term + linear_term


def sample_outcome_model_gp(X, U, param, seed):  # works with 2D covariates only

    np.random.seed(seed)
    XX, UU = np.meshgrid(X, U)
    XU_flat = np.c_[XX.ravel(), UU.ravel()]

    mean = np.zeros(len(XU_flat))

    if param["kernel"] == "rbf":
        K = rbf_linear_kernel(XU_flat, XU_flat, np.array(param["ls"]), np.array(param["alpha"]))

    f_sample = np.random.multivariate_normal(mean, K)
    Y = f_sample.reshape(XX.shape)

    gp_func = interpolate.interp2d(X, U, Y, kind="linear")

    return gp_func


def regression_model(df, regressors, target, model, params):
    lr = len(regressors)
    X = np.array(df[regressors]).reshape(-1, lr)
    y = np.array(df[target])

    if model == "poly":
        poly = PolynomialFeatures(degree=params["degree"], include_bias=False)

        if lr > 1:
            X_poly = np.hstack((poly.fit_transform(X[:,:-1].reshape(-1, lr - 1)), X[:,-1].reshape(-1,1)))
        else:
            X_poly = poly.fit_transform(X)

        model = RidgeCV(alphas=params["Cs"], cv=params["cv_folds"])
        model.fit(X_poly, y)

    elif model == "NN":
        model = MLPRegressor(hidden_layer_sizes=params["hls"], activation=params["act"], early_stopping=params["early_stp"], validation_fraction=params["val_frac"])
        model.fit(X, y)

    else:
        raise NotImplementedError(f'{model} for regression in the trial sample is not implemented')

    return model


def logistic_model(df, adj_covs, target, model, params):

    lr = len(adj_covs)
    X = np.array(df[adj_covs]).reshape(-1, lr)
    y = np.array(df[target])

    if model == "poly":
        poly = PolynomialFeatures(degree=params["degree"], include_bias=False)
        if lr > 1:
            X_poly = np.hstack((poly.fit_transform(X[:,:-1].reshape(-1, lr - 1)), X[:,-1].reshape(-1,1)))
        else:
            X_poly = poly.fit_transform(X,)

        model = LogisticRegressionCV(Cs=params["Cs"], penalty=params["penalty"], max_iter=100)
        model.fit(X_poly, y)

    else:
        raise NotImplementedError(f'{model} for regression in the trial sample is not implemented')

    return model


def get_om_fits(g_a_X, b_a_X, h_a_X, f_a_X, regressors, model, pdeg, X_test):

    if f_a_X == "noise":
        f_a_X_test = 5 * np.random.randn(len(X_test))
    else:
        f_a_X_test = f_a_X.predict(X_test.reshape(-1, 1))

    if model == "poly":
        poly = PolynomialFeatures(degree=pdeg, include_bias=False)
        X_test = poly.fit_transform(X_test.reshape(-1,len(regressors)))
        X_test_aug = np.hstack((X_test.reshape(-1, pdeg), f_a_X_test.reshape(-1, 1)))

    gax_test = g_a_X.predict(X_test)   
    bax_test = b_a_X.predict(X_test)
    hax_test = h_a_X.predict(X_test_aug) 

    return gax_test, bax_test, hax_test