#! /usr/bin/env python
from __future__ import division
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

ml_parameters = []
with open("parameters.txt","r") as f:
    next(f)
    for line in f:
        fields = line.split("\t")
        ml_parameters.append(float(fields[1]))

time = 73

### helper functions ###################################################################################################
def model_96well_tt(parameters):

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_96well]

    solution_96well = []
    for c in C_96well:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        V = np.log10((solution.T[N_E + 2]))
        solution_96well.append(V)

    return solution_96well


def model_6well_tt(parameters):

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_6well]

    solution_6well = []
    for c in C_6well:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        V = np.log10((solution.T[N_E + 2]))
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


### plot ###############################################################################################################
tt          = np.linspace(0, 97, 971)
tt_points   = [17, 25, 49, 73, 97]

dim         = 8
C_6well     = [0, 0.5, 5]
C_96well    = [0.01, 0.1, 0.2, 0.5, 2, 10]
epsilon_max = 1
V0_6well    = 1.6e+4
V0_96well   = 8e+3


### plot ###
fontsize = 12
markersize = 8
alpha = 0.5

labels = [r"(A) PB28 0.01 \textmu M, end-point infection",
          r"(B) PB28 0.1 \textmu M, end-point infection",
          r"(C) PB28 0.2 \textmu M, end-point infection",
          r"(D) PB28 0.5 \textmu M, end-point infection",
          r"(E) PB28 2 \textmu M, end-point infection",
          r"(F) PB28 10 \textmu M, end-point infection",
          r"(G) control, time-resolved infection",
          r"(H) PB28 0.5 \textmu M, time-resolved infection",
          r"(I) PB28 5 \textmu M, time-resolved infection"]

par_names = "beta", "tau_L", "tau_I", "p", "omega_0", "IC_50", "N_eps", "n_L"
par_order = "FigS5", "FigS6", "FigS3", "FigS4", "FigS7", "FigS9", "FigS10", "FigS8"

parameters_upper = [x + 0.5 * ml_parameters[i] for i, x in enumerate(ml_parameters)]
parameters_lower = [x - 0.5 * ml_parameters[i] for i, x in enumerate(ml_parameters)]

for aa, (par_low, par_upp) in enumerate(zip(parameters_lower,parameters_upper)):

    pars_lower = ml_parameters
    pars_upper = ml_parameters

    pars_lower = [parameters_lower[aa] if x == ml_parameters[aa] else x for x in pars_lower]
    pars_upper = [parameters_upper[aa] if x == ml_parameters[aa] else x for x in pars_upper]

    maxlik_solution_6well = model_6well_tt(ml_parameters)
    maxlik_solution_96well = model_96well_tt(ml_parameters)

    lower_solution_6well = model_6well_tt(pars_lower)
    lower_solution_96well = model_96well_tt(pars_lower)

    upper_solution_6well = model_6well_tt(pars_upper)
    upper_solution_96well = model_96well_tt(pars_upper)

    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    kk = -1
    for ii in range(3):
        for jj in range(3):
            kk = kk + 1
            axs[ii,jj].xaxis.set_ticks([0,17,25,49,73,97])
            axs[ii,jj].xaxis.set_ticklabels([0,17,25,49,73,97],fontsize=fontsize)
            axs[ii, jj].set_xlim(-3.0, 100.0)
            axs[ii,jj].yaxis.set_ticks([3, 4, 5, 6, 7])
            axs[ii,jj].yaxis.set_ticklabels([3, 4, 5, 6, 7],fontsize=fontsize)
            axs[ii, jj].set_ylim(3, 7)
            axs[ii, jj].text(0, 6.6, labels[kk],
                             fontsize=fontsize,
                             color='black')

            axs[ii,jj].set_xlabel("Time (hours post infection)", fontsize=fontsize),
            axs[ii,jj].set_ylabel(r"Viral load (log$_{10}$ PFU$_\mathrm{e}$/mL)", fontsize=fontsize)

            if kk <= 5:
                axs[ii, jj].plot(tt, maxlik_solution_96well[kk],
                                 color="black",  # lightgreen",#colors[kk],
                                 linestyle="-",
                                 linewidth=1,
                                 alpha=1,
                                 label="best-fit")
                axs[ii, jj].plot(tt, lower_solution_96well[kk],
                                 color="blue",
                                 linestyle="--",
                                 linewidth=1,
                                 alpha=1,
                                 label=r"$-50\%$")
                axs[ii, jj].plot(tt, upper_solution_96well[kk],
                                 color="red",
                                 linestyle="--",
                                 linewidth=1,
                                 alpha=1,
                                 label=r"$+50\%$")
                axs[ii, jj].legend(loc="lower right", fontsize=8)

            if kk > 5:
                mm = kk - 6
                axs[ii,jj].plot(tt, maxlik_solution_6well[mm],
                         color="black",
                         linestyle="-",
                         linewidth=1,
                         alpha=1,
                         label="best-fit")
                axs[ii,jj].plot(tt, lower_solution_6well[mm],
                         color="blue",
                         linestyle="--",
                         linewidth=1,
                         alpha=1,
                         label=r"$-50\%$")
                axs[ii,jj].plot(tt, upper_solution_6well[mm],
                         color="red",
                         linestyle="--",
                         linewidth=1,
                         alpha=1,
                         label=r"$+50\%$")
                axs[ii, jj].legend(loc="lower right",fontsize=8)

    fig.tight_layout()
    plt.savefig("./figures/" + par_order[aa] + ".pdf", format="pdf", transparent=True)
    plt.savefig("./figures/" + par_order[aa] + ".tiff", format="tiff")
    # plt.show()