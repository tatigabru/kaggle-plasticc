"""
10x times fasted implementation of CAR extractor compared to FEETs package 

"""
import warnings
import numpy as np
from scipy.optimize import minimize
from numba import jit


@jit(nopython=True)
def _car_like(parameters, t, x, error_vars):
    EPSILON = 1e-300
    CTE_NEG = -np.infty

    sigma, tau = parameters
    # t, x, error_vars = t.flatten(), x.flatten(), error_vars.flatten()

    m = np.mean(x)
    N = x.shape[0]
    Omega = np.zeros(N)
    Omega[0] = (tau * (sigma ** 2)) / 2.
    x_hat = np.zeros(N)

    x_ast = x - m
    A = np.exp(-(t[1:] - t[:-1]) / tau)

    loglik = 0.

    for i in range(1, N):

        x_hat[i] = A[i - 1] * x_hat[i - 1] + (A[i - 1] * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) * (
                x_ast[i - 1] - x_hat[i - 1])

        Omega[i] = Omega[0] * (1 - (A[i - 1] ** 2)) + (A[i - 1] ** 2) * Omega[i - 1] * (
                1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])))

        loglik_inter = np.log(
            ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
            (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
                            (Omega[i] + error_vars[i]))) + EPSILON))

        loglik = loglik + loglik_inter

        if loglik <= CTE_NEG:
            break

    return -loglik


def calculate_car(time, magnitude, error, minimize_method='nelder-mead'):
    e = error ** 2

    x0 = np.array([10, 0.5])
    bounds = ((0, 100), (0, 100))
    warnings.filterwarnings('ignore')
    res = minimize(_car_like, x0,
                   args=(time, magnitude, e),
                   method=minimize_method, bounds=bounds)
    sigma, tau = res.x[0], res.x[1]
    return sigma, tau
