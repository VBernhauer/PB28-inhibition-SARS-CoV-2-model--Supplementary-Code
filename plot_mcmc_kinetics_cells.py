#! /usr/bin/env python
from __future__ import division
from pickle import load


from scipy.integrate import odeint

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'Helvetica'

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

### viral load PB28 0uM ################################################################################################
### read data ###
file_virus_0 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 0uM")
t_file_virus = file_virus_0["Time"].values[:]

### inhibition PB28 0uM ################################################################################################
file_pb28_virus = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="virus inhibition PB28 at 72h")

### MCMC chains_old ####################################################################################################
with open('chains/chains.obj', 'rb') as a:
    chains = load(a)#, encoding='latin1')

with open('chains/logprob.obj', 'rb') as b:
    log_prob = load(b)#, encoding='latin1')

### helper functions ###################################################################################################
def model_96well_tt(parameters):

    logparameters = [10 ** x for idx, x in enumerate(parameters) if idx < 7]
    parameters = np.append(logparameters, parameters[7])

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_96well]

    solution_96well = []
    for c in C_96well:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        T = solution.T[0]
        E_ii = []
        for ii in range(N_E):
            E_ii.append(solution.T[ii + 1])
        E = np.sum(E_ii,axis=0)
        I = solution.T[N_E+ 1]
        solution_96well.append([T, E, I])

    return solution_96well


def model_6well_tt(parameters):

    logparameters = [10 ** x for idx, x in enumerate(parameters) if idx < 7]
    parameters = np.append(logparameters, parameters[7])

    N_E     = int(parameters[7])
    init    = [1] + N_E * [0] + [0] + [V0_6well]

    solution_6well = []
    for c in C_6well:
        eps = epsilon(parameters, c)
        solution = odeint(ode_kinetics, init, tt, args=(parameters, eps,), mxstep=5000)
        T = solution.T[0]
        E_ii = []
        for ii in range(N_E):
            E_ii.append(solution.T[ii + 1])
        E = np.sum(E_ii,axis=0)
        I = solution.T[N_E + 1]
        solution_6well.append([T, E, I])

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
C_96well    = file_pb28_virus["PB28 [uM]"].values
epsilon_max = 1
V0_6well    = 1.6e+4
V0_96well   = 8e+3


### index of the maximum likelihood value ##############################################################################
idx_max_likelihood = np.argmax(log_prob)
maxlik_solution_6well  = model_6well_tt(chains[idx_max_likelihood,:])
maxlik_solution_96well = model_96well_tt(chains[idx_max_likelihood,:])

### get the bounds on the solutions ####################################################################################
solutions_6well = []
solutions_96well = []
for i in range(int(np.size(chains)/dim)):
    # if i % 5 == 0:
    solutions_6well.append(model_6well_tt(chains[i,:]))
    solutions_96well.append(model_96well_tt(chains[i,:]))
solutions_6well = np.array(solutions_6well)
solutions_96well = np.array(solutions_96well)

solutions_6well_out  = [[],[],[]]
solutions_96well_out = [[],[],[],
                        [],[],[]]
for sample, concentration in enumerate(solutions_6well):
    for ii in range(len(concentration)):
        solutions_6well_out[ii].append(solutions_6well[sample][ii])

for sample, concentration in enumerate(solutions_96well):
    for ii in range(len(concentration)):
        solutions_96well_out[ii].append(solutions_96well[sample][ii])

solutions_6well_out  = np.array(solutions_6well_out)
solutions_96well_out = np.array(solutions_96well_out)

solution_min_6well = []
solution_max_6well = []

solution_min_96well = []
solution_max_96well = []


for c_idx, infection in enumerate(solutions_6well_out):
    solution_6well_infection = solutions_6well_out[c_idx]
    solution_min_6well_p = []
    solution_max_6well_p = []
    for p in range(np.shape(solution_6well_infection)[1]):
        solution_min_6well_t = []
        solution_max_6well_t = []
        for idx, t in enumerate(tt):
            # solution_min_6well_t_p = np.min(solution_6well_infection[:,p,idx])
            solution_min_6well_t_p = np.percentile(solution_6well_infection[:, p, idx], 2.5)
            solution_min_6well_t.append(solution_min_6well_t_p)

            # solution_max_6well_t_p = np.max(solution_6well_infection[:, p, idx])
            solution_max_6well_t_p = np.percentile(solution_6well_infection[:, p, idx], 97.5)
            solution_max_6well_t.append(solution_max_6well_t_p)

        solution_min_6well_p.append(solution_min_6well_t)
        solution_max_6well_p.append(solution_max_6well_t)

    solution_min_6well.append(solution_min_6well_p)
    solution_max_6well.append(solution_max_6well_p)

for c_idx, infection in enumerate(solutions_96well_out):
    solution_96well_infection = solutions_96well_out[c_idx]
    solution_min_96well_p = []
    solution_max_96well_p = []
    for p in range(np.shape(solution_96well_infection)[1]):
        solution_min_96well_t = []
        solution_max_96well_t = []
        for idx, t in enumerate(tt):
            # solution_min_96well_t_p = np.min(solution_96well_infection[:,p,idx])
            solution_min_96well_t_p = np.percentile(solution_96well_infection[:, p, idx], 2.5)
            solution_min_96well_t.append(solution_min_96well_t_p)

            # solution_max_96well_t_p = np.max(solution_96well_infection[:, p, idx])
            solution_max_96well_t_p = np.percentile(solution_96well_infection[:, p, idx], 97.5)
            solution_max_96well_t.append(solution_max_96well_t_p)

        solution_min_96well_p.append(solution_min_96well_t)
        solution_max_96well_p.append(solution_max_96well_t)

    solution_min_96well.append(solution_min_96well_p)
    solution_max_96well.append(solution_max_96well_p)


### plot ###
fontsize = 12
markersize = 6
alpha = 0.35
labels = [r"(A) PB28 0.01 \textmu M, end-point infection",
          r"(B) PB28 0.1 \textmu M, end-point infection",
          r"(C) PB28 0.2 \textmu M, end-point infection",
          r"(D) PB28 0.5 \textmu M, end-point infection",
          r"(E) PB28 2 \textmu M, end-point infection",
          r"(F) PB28 10 \textmu M, end-point infection",
          r"(G) control, time-resolved infection",
          r"(H) PB28 0.5 \textmu M, time-resolved infection",
          r"(I) PB28 5 \textmu M, time-resolved infection"]

colors = ["green",
          "orange",
          "red"]
population = [r"susceptible ($S$)",
              r"latent ($L$)",
              r"infectious ($I$)"]

fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

kk = -1
for ii in range(3):
    for jj in range(3):
        kk = kk + 1
        axs[ii, jj].xaxis.set_ticks(np.append([0], t_file_virus))
        axs[ii, jj].xaxis.set_ticklabels(np.append([0], t_file_virus), fontsize=fontsize)
        axs[ii, jj].set_xlim(-3.0, 100.0)
        # if kk <= 5:
        axs[ii, jj].yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[ii, jj].yaxis.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=fontsize)
        axs[ii, jj].set_ylim(-0.1, 1.15)
        axs[ii, jj].text(0, 1.025, labels[kk],
                         fontsize=fontsize,
                         color='black')

        axs[ii, jj].set_xlabel("Time (hours post infection)", fontsize=fontsize),
        axs[ii, jj].set_ylabel("Fraction of cells per well", fontsize=fontsize)

        if kk <= 5:
            for nn in range(3):
                # axs[ii, jj].fill_between(tt, solution_min_96well[kk][nn], solution_max_96well[kk][nn], facecolor=colors[nn],
                #                                                                                  edgecolor=colors[nn],
                #                                                                                  alpha=alpha)
                axs[ii,jj].plot(tt, maxlik_solution_96well[kk][nn],
                         color=colors[nn],
                         linestyle="-",
                         linewidth=1,
                         alpha=1)
                axs[ii,jj].plot(tt, solution_min_96well[kk][nn],
                         color=colors[nn],
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)
                axs[ii,jj].plot(tt, solution_max_96well[kk][nn],
                         color=colors[nn],
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)
        if kk > 5:
            mm = kk - 6
            for nn in range(3):
                # axs[ii,jj].fill_between(tt, solution_min_6well[mm][nn], solution_max_6well[mm][nn], facecolor=colors[nn],
                #                                                                             edgecolor=colors[nn],
                #                                                                             alpha=alpha,
                #                                                                             label=population[nn])
                axs[ii,jj].plot(tt, maxlik_solution_6well[mm][nn],
                         color=colors[nn],
                         linestyle="-",
                         linewidth=1,
                         alpha=1)
                axs[ii,jj].plot(tt, solution_min_6well[mm][nn],
                         color=colors[nn],
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)
                axs[ii,jj].plot(tt, solution_max_6well[mm][nn],
                         color=colors[nn],
                         linestyle="--",
                         linewidth=1,
                         alpha=0.5)

mp = [[], [], []]
for ii in range(len(population)):
    mp[ii] = mpatches.Patch(color=colors[ii], alpha=alpha, linewidth=0)
# handles, labels = axs[0, 1].get_legend_handles_labels()
for ii in range(3):
    for jj in range(3):
        axs[ii, jj].legend(mp, population, loc='center left',
                           fontsize=8)
# axs[0, 1].legend(mp, population, ncol=len(population),
#                                  columnspacing=0.5,
#                                  loc='upper center',
#                                  bbox_to_anchor=(0.4, 1.25))

fig.tight_layout()
plt.savefig("../LaTeX/figures/kinetics_cells.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/kinetics_cells.tiff", format="tiff")
plt.show()