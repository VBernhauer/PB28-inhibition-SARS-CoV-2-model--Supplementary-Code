#! /usr/bin/env python
from __future__ import division
from pickle import dump

from scipy.integrate import odeint
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
import os
import emcee
import time

chains = "./chains"
if not os.path.exists(chains):
    os.makedirs(chains)

### viral load PB28 0uM ################################################################################################
### read data ###
file_virus_0 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 0uM")
t_file_virus = file_virus_0["Time"].values[:]
data_file_virus_0 = np.log10(file_virus_0.values.T[1:4])
std_data_file_virus_0 = np.mean(np.std(data_file_virus_0, axis=0, ddof=1))

### viral load PB28 0.5uM ##############################################################################################
### read data ###
file_virus_05 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 0.5uM")
data_file_virus_05 = np.log10(file_virus_05.values.T[1:4])
std_data_file_virus_05 = np.mean(np.std(data_file_virus_05, axis=0, ddof=1))

### viral load PB28 5uM ################################################################################################
### read data ###
file_virus_5 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 5uM")
data_file_virus_5 = np.log10(file_virus_5.values.T[1:4])
std_data_file_virus_5 = np.mean(np.std(data_file_virus_5, axis=0, ddof=1))

### inhibition PB28  ###################################################################################################
### read data ###
file_pb28_virus = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="virus inhibition PB28 at 72h")
data_virus_pb28 = np.log10(file_pb28_virus.values.T[1:4])
std_data_virus_pb28 = np.mean(np.std(data_virus_pb28,axis=0,ddof=1))

data_virus = [data_file_virus_0, data_file_virus_05, data_file_virus_5]
std_data_virus = [std_data_file_virus_0, std_data_file_virus_05, std_data_file_virus_5]

r_pb28 = np.shape(data_virus_pb28)[0]
r_tcourse = len(data_virus)

########################################################################################################################
### log-likelihood formula ###
def lognormpdf(data, model, stdev):

    return - 0.5 * ((data - model) / stdev) ** 2 - np.log(np.sqrt(2 * np.pi) * stdev)


### log-probability ###
def logprob(parameters):

    logparameters = [10 ** x for idx, x in enumerate(parameters) if idx < 7]
    parameters = np.append(logparameters, parameters[7])

    lp = logprior(parameters)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + loglike(parameters)


### priors ###
def logprior(parameters):

    if (not all(x >= bound_lower[id_lower] for id_lower, x in enumerate(parameters)) or
        not all(x <= bound_upper[id_upper] for id_upper, x in enumerate(parameters))):
        return -np.inf

    return 0.0


### log-likelihood ###
def loglike(parameters):

    solution_virus = model_6well(parameters)
    value = 0
    for ii in range(0, len(data_virus)):
        for jj in range(0, r_tcourse):
            value = value + np.sum(lognormpdf(data_virus[ii][jj], solution_virus[ii], std_data_virus[ii]))


    solution_virus_pb28 = model_96well(parameters)
    for ii in range(0, r_pb28):
        value = value + np.sum(lognormpdf(data_virus_pb28[ii], solution_virus_pb28, std_data_virus_pb28))

    if np.isnan(value):
        print(parameters)

    return value


def model_96well(parameters):

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_96well]

    solution_96well = []
    for c in C:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        V = np.log10((solution.T[N_E + 2])[get_time_id[3]])
        solution_96well.append(V)

    return solution_96well


def model_6well(parameters):

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_6well]

    solution_6well = []
    for c in C_6well:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        V = np.log10((solution.T[N_E + 2])[get_time_id])
        solution_6well.append(V)

    return solution_6well


def ode_kinetics(state, t, parameters, eps):

    w0 = parameters[4]
    omega = washing(t, w0)

    N_E = int(parameters[7])

    T = state[0]
    E = state[1:N_E + 1]
    I = state[N_E + 1]
    V = state[N_E + 2]

    dE = [0 for i in range(N_E)]

    dT = - parameters[0] * T * V
    dE[0] = parameters[0] * T * V - N_E / parameters[1] * E[0]
    for i in range(1, len(E), 1):
        dE[i] = N_E / parameters[1] * (E[i - 1] - E[i])
    dI = N_E / parameters[1] * E[N_E - 1] - 1 / parameters[2] * I
    dV = (1 - eps) * parameters[3] * I - omega * V

    ode_solution = [dT] + dE + [dI] + [dV]

    return ode_solution


def washing(t, w0):
    # w0 ... strength of washing
    # wd ... standard deviation of the length of washing (hours)
    # wt ... time of washing implementation (hours)

    wd = 0.05
    wt = 1

    w = w0 * 1 / np.sqrt(2 * np.pi * wd ** 2) * np.exp(-(t - wt) ** 2 / (2 * wd ** 2))

    return w


def epsilon(parameters,c):

    eps = epsilon_max * c ** parameters[6] / (parameters[5] ** parameters[6] + c ** parameters[6])

    return eps


########################################################################################################################
tt = np.linspace(0, 97, 971)
tt_points = [17, 25, 49, 73, 97]
get_time_id = []
for tt_item in tt_points:
    for t_id, t in enumerate(tt):
        if t == tt_item:
            get_time_id.append(t_id)

C_6well     = [0, 0.5, 5]
C           = file_pb28_virus["PB28 [uM]"].values
epsilon_max = 1
V0_6well    = 1.6e+4
V0_96well   = 8e+3

### bounds #############################################################################################################
beta_lower          = 0
beta_upper          = np.inf
tau_E_lower         = 0
tau_E_upper         = np.inf
tau_I_lower         = 1
tau_I_upper         = 100
p_lower             = 0
p_upper             = np.inf
omega_0_lower       = 0
omega_0_upper       = np.inf
IC50_lower          = 0
IC50_upper          = np.inf
N_eps_lower         = 0
N_eps_upper         = np.inf
n_E_lower           = 1
n_E_upper           = 50

bound_lower = [beta_lower,
               tau_E_lower,
               tau_I_lower,
               p_lower,
               omega_0_lower,
               IC50_lower,
               N_eps_lower,
               n_E_lower]
bound_upper = [beta_upper,
               tau_E_upper,
               tau_I_upper,
               p_upper,
               omega_0_upper,
               IC50_upper,
               N_eps_upper,
               n_E_upper]

# virus parameters
beta        = np.log10(5e-7)
tau_L       = np.log10(2e+1)
tau_I       = np.log10(1e+1)
p           = np.log10(1e+5)
omega_0     = np.log10(1.2e+0)
IC50        = np.log10(5e-1)
N_eps       = np.log10(1)
n_E         = 20

parameters_init = beta, tau_L, tau_I, p, omega_0, IC50, N_eps, n_E

#### MCMC #############################################################################################################
Nsamples = 20000  # number of final posterior samples
pos = parameters_init + 1e-4 * np.random.randn(2 * len(parameters_init), len(parameters_init))
nwalkers, ndim = pos.shape


def main():
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)

        start = time.time()
        sampler.run_mcmc(pos, Nsamples, progress=True)
        end = time.time()
        multi_time = end - start

        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    chains_trace = sampler.chain

    flat_chains = sampler.get_chain(discard=int(Nsamples / 2), thin=10, flat=True)
    flat_logprob = sampler.get_log_prob(discard=int(Nsamples / 2), thin=10, flat=True)

    with open('chains/chains_trace.obj', 'wb') as a:
        dump(chains_trace, a)
    with open('chains/logprob.obj', 'wb') as b:
        dump(flat_logprob, b)
    with open('chains/chains.obj', 'wb') as c:
        dump(flat_chains, c)


if __name__ == "__main__":
    freeze_support()
    main()
