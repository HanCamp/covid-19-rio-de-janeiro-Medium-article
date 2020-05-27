import numpy as np
import math
import copy
import sys
from time import time
from numba import jit
from numpy.linalg import norm
from scipy.integrate import odeint
from matplotlib.pyplot import hist, show

def seasonal_beta(t, beta_prime, mag):
    return beta_prime * (1 - (mag / 2) * (1 - math.cos(2 * math.pi * t / 365)))

def kinetic_beta(t, beta0, tau_beta):
    return beta0 * math.exp(- t / tau_beta)

def kinetic_gamma(t, gamma0, gamma1, tau_gamma):
    return gamma0 + gamma1 / (1 + math.exp(-t + tau_gamma))

def kinetic_mu(t, mu0, mu1, tau_mu):
    return mu0 * math.exp(-t / tau_mu) + mu1

@jit
def SIRD_kinetic_seasonal_ODE(xs, t, params):
    """ Standard Seasonal Kinetic SIRD model """
    beta0, beta0_mag, tau_beta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu = params
    xs_grad = np.empty(xs.shape)
    beta = kinetic_beta(t, beta0, tau_beta) * (1 + beta0_mag * ( 1 - math.cos(2 * math.pi * t / 365)) / 2)
    gamma = kinetic_gamma(t, gamma0, gamma1, tau_gamma)
    mu = kinetic_mu(t, mu0, mu1, tau_mu)
    N = xs[0] + xs[1] + xs[2]
    xs_grad[0] = -beta * xs[1] * xs[0] / N
    xs_grad[1] = beta * xs[1] * xs[0] / N - (gamma + mu) * xs[1]
    xs_grad[2] = gamma * xs[1]
    xs_grad[3] = mu * xs[1]
    return xs_grad

@jit
def SIRD_kinetic_seasonal_reactive_ODE(xs, t, params):
    """ Seasonal Kinetic SIRD model with reactive authorities"""
    delta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu = params
    xs_grad = np.empty(xs.shape)
    N = xs[0] + xs[1] + xs[2]
    gamma = kinetic_gamma(t, gamma0, gamma1, tau_gamma)
    mu = kinetic_mu(t, mu0, mu1, tau_mu)
    beta = (1 - delta) * (gamma + mu) * N / xs[0] + math.pow(10.0,-8.0)
    xs_grad[0] = -beta * xs[1] * xs[0] / N
    xs_grad[1] = beta * xs[1] * xs[0] / N - (gamma + mu) * xs[1]
    xs_grad[2] = gamma * xs[1]
    xs_grad[3] = mu * xs[1]
    return xs_grad

@jit
def SIRD_kinetic_seasonal_no_SD_ODE(xs, t, params):
    """ Seasonal Kinetic SIRD model without social distancing"""
    beta0, beta0_mag, tau_beta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu = params
    xs_grad = np.empty(xs.shape)
    beta = beta0 * (1 + beta0_mag * ( 1 - math.cos(2 * math.pi * t / 365)) / 2)
    gamma = kinetic_gamma(t, gamma0, gamma1, tau_gamma)
    mu = kinetic_mu(t, mu0, mu1, tau_mu)
    N = xs[0] + xs[1] + xs[2]
    xs_grad[0] = -beta * xs[1] * xs[0] / N
    xs_grad[1] = beta * xs[1] * xs[0] / N - (gamma + mu) * xs[1]
    xs_grad[2] = gamma * xs[1]
    xs_grad[3] = mu * xs[1]
    return xs_grad


def SIRD_kinetic_seasonal(t, x0, beta0, beta0_mag, tau_beta, gamma_sum, gamma_partition, tau_gamma, mu_sum, mu_partition, tau_mu):
    """
    Function used for making Seasonal Kinetic SIRD lmfit model
    """
    gamma0 = gamma_partition * gamma_sum
    gamma1 = (1 - gamma_partition) * gamma_sum
    mu0 = mu_partition * mu_sum
    mu1 = (1 - mu_partition) * mu_sum
    params = [beta0, beta0_mag, tau_beta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu]
    x_sol = odeint(SIRD_kinetic_seasonal_ODE, x0, t, args=(params,))
    return x_sol

def SIRD_kinetic_seasonal_reactive(t, x0, delta, gamma_sum, gamma_partition, tau_gamma, mu_sum, mu_partition, tau_mu):
    """
    Function used for making Seasonal Kinetic SIRD lmfit model in a scenario
    where authorities keep infection rate within a margin of maximum for virus
    suppression
    """
    gamma0 = gamma_partition * gamma_sum
    gamma1 = (1 - gamma_partition) * gamma_sum
    mu0 = mu_partition * mu_sum
    mu1 = (1 - mu_partition) * mu_sum
    params = [delta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu]
    x_sol = odeint(SIRD_kinetic_seasonal_reactive_ODE, x0, t, args=(params,))
    return x_sol

def SIRD_kinetic_seasonal_no_SD(t, x0, beta0, beta0_mag, tau_beta, gamma_sum, gamma_partition, tau_gamma, mu_sum, mu_partition, tau_mu):
    """
    Function used for making Seasonal Kinetic SIRD lmfit model in a scenario
    without any social distancing
    """
    gamma0 = gamma_partition * gamma_sum
    gamma1 = (1 - gamma_partition) * gamma_sum
    mu0 = mu_partition * mu_sum
    mu1 = (1 - mu_partition) * mu_sum
    params = [beta0, beta0_mag, tau_beta, gamma0, gamma1, tau_gamma, mu0, mu1, tau_mu]
    x_sol = odeint(SIRD_kinetic_seasonal_no_SD_ODE, x0, t, args=(params,))
    return x_sol

@jit
def H_loss(res, k):
    """ Bisquare loss function """
    H = np.empty(res.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if k[j] > math.sqrt(sys.float_info.max):
                K = math.sqrt(sys.float_info.max)
            else:
                K = k[j]
            if abs(res[i,j]) > K:
                H[i,j] = K**2 / 6
            else:
                H[i,j] = (K**2 / 6) * (1 - (1 - (abs(res[i,j]) / K)**2)**3)
            if H[i,j] == np.nan or H[i,j] == np.inf:
                exit(1)
    return H

@jit
def loss_function(params, t, x0, model=None, data=None):
    """ Weighed bisquare loss function """
    indxs = [1, 2, 3]
    x_pred = model.eval(params, t=t, x0=x0)
    x_pred_diff = np.diff(x_pred[:,indxs], axis=0)
    x_pred_diff[x_pred_diff < 0] = 0
    data_diff = np.diff(data[:,indxs], axis=0)
    data_diff[data_diff < 0] = 0
    res = x_pred_diff - data_diff
    res_MAD = np.mean(np.abs(res - np.mean(res, axis=0)), axis=0)
    k = 4.685 * res_MAD
    weights = np.max(np.abs(res), axis=0)
    weights = np.max(weights) / weights
    loss = H_loss(res, k).dot(weights)
    params.pretty_print()
    return loss
