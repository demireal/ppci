import numpy as np
from scipy import interpolate
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x, beta):
    return 1 / (1 + np.exp(- x @ beta))


def rbf_linear_kernel(X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1])):
    distances = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / length_scales, axis=2)
    rbf_term = np.exp(-0.5 * distances**2)
    linear_term = np.dot(np.dot(X1, np.diag(alpha)), X2.T)
    return rbf_term + linear_term


def sample_outcome_model_gp(param, X, U, seed):

    np.random.seed(seed)
    XX, UU = np.meshgrid(X, U)
    XU_flat = np.c_[XX.ravel(), UU.ravel()]

    K = rbf_linear_kernel(XU_flat, XU_flat, np.array(param['ls']), np.array(param['alpha']))
    mean = np.zeros(len(XU_flat))

    f_sample = np.random.multivariate_normal(mean, K)
    Y = f_sample.reshape(XX.shape)

    gp_func = interpolate.interp2d(X, U, Y, kind='linear')

    return gp_func


def fit_obs_outcome_fn(df_obs, regressors="X", target="Y", model="NN", hls=(128,32,8), activation="tanh"):
    df = df_obs.query("A==1").copy()
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    y = np.array(df[target])

    if model == "NN":
        model = MLPRegressor(hidden_layer_sizes=hls, activation=activation, early_stopping=True, validation_fraction=0.2)
        model.fit(X, y)
    else:
        raise NotImplementedError(f'{model} to fit f^a(X) is not implemented')

    return model


def fit_trial_outcome_fn(df_comp, regressors=["X", "fa(X)"], target="Y", model="linear", hls=(128,32,8), activation="tanh", poly_degree=10):
    df = df_comp.query("S==1 & A==1").copy()
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    y = np.array(df[target])

    if model == "NN":
        model = MLPRegressor(hidden_layer_sizes=hls, activation=activation, early_stopping=True, validation_fraction=0.1)
        model.fit(X, y)
    elif model== "linear":
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
    elif model== "poly":
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        if len(regressors) > 1:
            X_poly = np.hstack((poly.fit_transform(X[:,0].reshape(-1,1)), X[:,1].reshape(-1,1)))
        else:
            X_poly = poly.fit_transform(X)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_poly, y)
    else:
        raise NotImplementedError(f'{model} to fit h^a(tilde_X) is not implemented')

    return model


def fit_trial_bias_fn(df_comp, regressors="X", target="Z", model="linear", hls=(128,32,8), activation="tanh", poly_degree=10):
    df = df_comp.query("S==1 & A==1").copy()
    X = np.array(df[regressors]).reshape(-1,len(regressors))
    z = np.array(df[target])

    if model == "NN":
        model = MLPRegressor(hidden_layer_sizes=hls, activation=activation, early_stopping=True, validation_fraction=0.1)
        model.fit(X, z)
    elif model== "linear":
        model = LinearRegression(fit_intercept=True)
        model.fit(X, z)
    elif model== "poly":
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_poly, z)
    else:
        raise NotImplementedError(f'{model} to fit b^a(X) is not implemented')

    return model