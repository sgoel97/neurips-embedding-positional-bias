import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def get_r2(x: list | np.ndarray | pd.Series, y: list | np.ndarray | pd.Series):
    lr = LinearRegression()
    X, y = np.array(x).T, np.array(y)
    lr.fit(X, y)
    return lr.score(X, y)


def get_regression_coefs(x: list | np.ndarray | pd.Series, y: list | np.ndarray | pd.Series):
    lr = LinearRegression()
    X, y = np.array(x).T, np.array(y)
    lr.fit(X, y)
    return lr.coef_ / np.linalg.norm(lr.coef_)
