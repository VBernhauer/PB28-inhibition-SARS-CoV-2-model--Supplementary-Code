#! /usr/bin/env python
from __future__ import division
from pickle import load, dump
from multiprocessing import Pool, freeze_support
from scipy.integrate import odeint

import numpy as np
import os
import time

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./IC50"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def model_6well_tt(parameters, c):

    logparameters = [10 ** x for idx, x in enumerate(parameters) if idx < 7]
    parameters = np.append(logparameters, parameters[7])

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_6well]

    eps = epsilon(parameters, c)
    solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
    logV = np.log10((solution.T[N_E + 2])[get_time_id])

    return logV


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



### MCMC chains_old ####################################################################################################
with open('chains/chains.obj', 'rb') as a:
    chains = load(a)#, encoding='latin1')

dim         = 8
tt          = np.linspace(0, 97, 971)
tt_points   = [17, 25, 37, 49, 73, 97]
get_time_id = []
for tt_item in tt_points:
    for t_id, t in enumerate(tt):
        if t == tt_item:
            get_time_id.append(t_id)


epsilon_max = 1
V0_6well    = 1.6e+4
cc          = np.concatenate(([round(x, 2) for x in np.linspace(0.01, 0.9, 90)], [round(x, 1) for x in np.linspace(1, 10, 91)]))

### IC50 calculations ##################################################################################################
def output(c):

    solutions_6well = []
    for i in range(int(np.size(chains) / dim)):
        solution = model_6well_tt(chains[i, :], c)
        solutions_6well.append(solution)

    with open('IC50/IC50_' + str(c) + '.obj', 'wb') as sol:
        dump(solutions_6well, sol)

    print('Concentration C = ', str(c), ' finished.')

    return solutions_6well

def main():
    with Pool() as pool:
        start = time.time()
        pool.map(output, cc)
        end = time.time()
        multi_time = end - start

        print("Multiprocessing took {0:.1f} seconds".format(multi_time))


if __name__ == "__main__":
    freeze_support()
    main()
