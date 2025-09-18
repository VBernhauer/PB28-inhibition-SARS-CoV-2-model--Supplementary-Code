#! /usr/bin/env python
from __future__ import division
from pickle import load
from scipy.optimize import least_squares
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

### MCMC chains_old ####################################################################################################
with open('chains/chains.obj', 'rb') as a:
    chains = load(a)#, encoding='latin1')

with open('chains/logprob.obj', 'rb') as b:
    log_prob = load(b)#, encoding='latin1')


def model_6well_tt(parameters, c, time_id):

    logparameters = [10 ** x for idx, x in enumerate(parameters) if idx < 7]
    parameters = np.append(logparameters, parameters[7])

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_6well]

    eps = epsilon(parameters, c)
    solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
    V = np.log10((solution.T[N_E + 2])[time_id])

    return V


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


def virus(c, parameters):
    V = parameters[0] + (parameters[1] - parameters[0]) / (1 + 10 ** (parameters[3] * (np.log10(c) - np.log10(parameters[2]))))
    logV = np.log10(V)
    return logV


def residuals_virus(parameters, c, data):
    res = data - virus(c, parameters)
    return res


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
C = [0.01, 0.1, 0.2, 0.5, 2, 5, 10]


### plot ###############################################################################################################
fontsize = 12
markersize = 6
alpha = 0.35
alpha_data = 1
labels = [r"(A) 17 h post-infection",
          r"(B) 25 h post-infection",
          r"(C) 37 h post-infection",
          r"(D) 49 h post-infection",
          r"(E) 73 h post-infection",
          r"(F) 97 h post-infection"]

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

kk = -1
for ii in range(2):
    for jj in range(3):
        kk = kk + 1
        axs[ii,jj].xaxis.set_ticks(np.log10(C))
        axs[ii,jj].xaxis.set_ticklabels(["0.01", "0.1", "0.2", "0.5", "2", "5", "10"],fontsize=fontsize)
        axs[ii, jj].set_xlim(-2.1, 1.1)
        axs[ii,jj].yaxis.set_ticks([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
        axs[ii,jj].yaxis.set_ticklabels(["3.5", "4", "4.5", "5", "5.5", "6", "6.5"],fontsize=fontsize)
        axs[ii, jj].set_ylim(3.3, 6.7)
        axs[ii, jj].text(-2, 6.4, labels[kk],
                         fontsize=fontsize,
                         color='black')
        axs[ii,jj].set_xlabel(r"PB28 concentration (\textmu M)", fontsize=fontsize),
        axs[ii,jj].set_ylabel(r"Viral load (log$_{10}$ PFU$_\mathrm{e}$/mL)", fontsize=fontsize)

        ### IC50 calculations ##########################################################################################
        idx_max_likelihood = np.argmax(log_prob)
        time_point_id = get_time_id[kk]
        solution_6well_t_min = []
        solution_6well_t_max = []
        maxlik_solutions_6well_t = []
        for c in cc:
            maxlik_solution_6well = model_6well_tt(chains[idx_max_likelihood, :], c, time_point_id)
            maxlik_solutions_6well_t.append(maxlik_solution_6well)

            with open('./IC50//IC50_' + str(c) + '.obj', 'rb') as a:
                solution_6well_t_out = load(a)

            solution_6well_t = np.asarray(solution_6well_t_out)
            solutions_6well_min = np.percentile(solution_6well_t[:,kk], 2.5)
            solutions_6well_max = np.percentile(solution_6well_t[:,kk], 97.5)
            # solutions_6well_min = np.min(solution_6well_t[:,kk])
            # solutions_6well_max = np.max(solution_6well_t[:,kk])

            solution_6well_t_min.append(solutions_6well_min)
            solution_6well_t_max.append(solutions_6well_max)


        ### fit Hill to viral loads
        guess = np.array([10 ** maxlik_solutions_6well_t[-1], 10 ** maxlik_solutions_6well_t[0], 0.5, 2])
        opt_parameters = least_squares(residuals_virus, guess, args=(cc, maxlik_solutions_6well_t))
        IC50_parameters = opt_parameters.x

        # axs[ii,jj].fill_between(np.log10(cc), solution_6well_t_min, solution_6well_t_max, facecolor="grey",
        #                                                                                     edgecolor="grey",
        #                                                                                     alpha=alpha)

        axs[ii, jj].plot(np.log10(cc), maxlik_solutions_6well_t,
                         color="green",
                         linestyle="-",
                         linewidth=1,
                         alpha=1)
        axs[ii, jj].plot(np.log10(cc), solution_6well_t_min,
                         color="green",
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)
        axs[ii, jj].plot(np.log10(cc), solution_6well_t_max,
                         color="green",
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)
        axs[ii, jj].plot(np.log10(IC50_parameters[2]), virus(IC50_parameters[2], IC50_parameters),
                         marker="o",
                         markersize=markersize,
                         markeredgecolor="black",
                         markerfacecolor="green",
                         alpha=1)
        axs[ii, jj].plot(np.log10(IC50_parameters[2]) * np.ones(10), np.linspace(3.3, virus(IC50_parameters[2], IC50_parameters),10),
                    color="black",
                    linestyle=":",
                    alpha=0.5)
        axs[ii, jj].plot(np.linspace(-2.1, np.log10(IC50_parameters[2]), 10), np.ones(10) * virus(IC50_parameters[2], IC50_parameters),
                    color="black",
                    linestyle=":",
                    alpha=0.5)
        # axs[ii, jj].plot(np.log10(cc), virus(cc, IC50_parameters),
        #                  color="red",
        #                  linestyle="-",
        #                  linewidth=1,
        #                  alpha=alpha)
        axs[ii, jj].text(-0.4, 3.5, "IC$_{50}$ = " + str(round(IC50_parameters[2], 3)) + r" \textmu M",
                         fontsize=fontsize,
                         color="green")

        print("Time t = " + str(tt_points[kk]) + " h done.")

fig.tight_layout()
plt.savefig("../LaTeX/figures/inhibition_IC50_timelaps.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/inhibition_IC50_timelaps.tiff", format="tiff")
plt.show()