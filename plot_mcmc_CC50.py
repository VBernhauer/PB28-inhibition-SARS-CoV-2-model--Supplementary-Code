#! /usr/bin/env python
from __future__ import division
from pickle import dump
from scipy.integrate import odeint
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import emcee
import seaborn as sns

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

### cytotoxicity PB28 0uM ################################################################################################
### read data ###
file = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="cell viability PB28 at 72h")
C = file["PB28 [uM]"].values[:]
data = file.values.T[1:6]
std_data = np.mean(np.std(data,axis=0,ddof=1))

########################################################################################################################
### log-likelihood formula ###
def lognormpdf(data, model, stdev):
    return - 0.5 * ((data - model) / stdev) ** 2 - np.log(np.sqrt(2 * np.pi) * stdev)


### log-probability ###
def logprob(parameters):
    # parameters = [10 ** x for x in parameters]
    lp = logprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(parameters)


### priors ###
def logprior(parameters):
    if not all(x > 0 for x in parameters):
        return -np.inf
    return 0.0


### log-likelihood ###
def loglike(parameters):
    fraction = cytotoxicity(parameters, C)
    value = 0
    for ii in range(np.size(data,0)):
            value = value + np.sum(lognormpdf(data[ii],fraction,std_data))
    return value


def cytotoxicity(parameters, c):
    fraction = parameters[0] / (1 + 10 ** (parameters[2] * (np.log10(c) - np.log10(parameters[1]))))
    return fraction


def residuals_cells(parameters, c, data):
    res = data - cytotoxicity(parameters, c)
    return res

########################################################################################################################
fraction_max_lower  = 0
fraction_max_upper  = np.inf
CC_50_lower   = 0
CC_50_upper   = np.inf
N_lower       = 0
N_upper       = np.inf

bound_lower = [fraction_max_lower,
               CC_50_lower,
               N_lower]
bound_upper = [fraction_max_upper,
               CC_50_upper,
               N_upper]

fraction_max    = np.log10(100)
CC50            = np.log10(4e+0)
N               = np.log10(1e+1)

parameters = [fraction_max, CC50, N ]

chains_folder = "./chains_CC50"
if not os.path.exists(chains_folder):
    os.makedirs(chains_folder)

#### MCMC growth ######################################################################################################
pos = 10 ** (parameters + 1e-4 * np.random.randn(2 * len(parameters), len(parameters)))
nwalkers, ndim = np.shape(pos)

sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
Nsamples = 20000  # number of final posterior samples

print("Processing...")
sampler.run_mcmc(pos, Nsamples, progress=True)

chains = sampler.get_chain(discard=int(Nsamples/2), thin=10, flat=True)
log_prob = sampler.get_log_prob(discard=int(Nsamples/2), thin=10, flat=True)

with open('chains_CC50/logprob.obj', 'wb') as a:
    dump(log_prob, a)
with open('chains_CC50/chains.obj', 'wb') as b:
    dump(chains, b)

cc = np.concatenate([np.linspace(0.01, 0.99, 99), np.linspace(1, 10, 91)])

### plot cytotoxicity - growth #########################################################################################
### index of the maximum likelihood value ##############################################################################
idx_max_likelihood = np.argmax(log_prob)
maxlik_parameters = chains[idx_max_likelihood,:]
maxlik_solution = cytotoxicity(chains[idx_max_likelihood,:],cc)

### get the bounds on the solutions ####################################################################################
solution = []
for i in range(int(np.size(chains)/ndim)):
    solution.append(cytotoxicity(chains[i,:],cc))
solution = np.asarray(solution)

solution_min = []
solution_max = []
for idx, concentration in enumerate(cc):
    solution_min_idx = np.percentile(solution[:,idx],2.5)
    solution_min.append(solution_min_idx)
    solution_max_idx = np.percentile(solution[:,idx], 97.5)
    solution_max.append(solution_max_idx)
    # solution_min_idx = np.min(solution[:,idx])
    # solution_min.append(solution_min_idx)
    # solution_max_idx = np.max(solution[:,idx])
    # solution_max.append(solution_max_idx)


fig = plt.figure(figsize=(5, 4))

markersize = 8
capsize = 8
alpha = 1
linewidth = 1.5
elinewidth = 1.5
fontsize = 12

frame = fig.gca()
frame.axes.xaxis.set_ticks(np.log10(C))
frame.axes.xaxis.set_ticklabels(["0.01", "0.1", "0.2", "0.5", "2", "10"], fontsize=fontsize)
frame.axes.yaxis.set_ticks([70, 80, 90, 100, 110, 120])
frame.axes.yaxis.set_ticklabels(["70", "80", "90", "100", "110", "120"], fontsize=fontsize)
plt.ylim(70, 120)
plt.xlabel(r"PB28 concentration (\textmu M)", fontsize=fontsize)
plt.ylabel(r"A549-ACE2 cell viability relative to control (\%)", fontsize=fontsize)
# plt.fill_between(np.log10(cc), solution_min, solution_max, facecolor="red", edgecolor="red", alpha=0.2)
plt.plot(np.log10(cc), maxlik_solution, color="red",
                                         linestyle="-",
                                         linewidth=1,
                                         alpha=1)
plt.plot(np.log10(cc), solution_min, color="red",
                                         linestyle="--",
                                         linewidth=1,
                                         alpha=0.5)
plt.plot(np.log10(cc), solution_max, color="red",
                                         linestyle="--",
                                         linewidth=1,
                                         alpha=0.5)
for line in data:
    plt.plot(np.log10(C), line, marker="o",
                                        color="red",
                                        markeredgecolor="black",
                                        linestyle=" ",
                                        markersize=markersize,
                                        alpha=alpha)
plt.text(-2, 73, "CC$_{50}$ = " + str(round(maxlik_parameters[1], 3)) + r" \textmu M",
               fontsize=fontsize,
               color='red')#,
               # weight='bold')
# plt.text(-2, 115, "(A)",
#                fontsize=14,
#                color='black')
fig.tight_layout()
plt.savefig("../LaTeX/figures/cytotoxicity.pdf", format="pdf", transparent=True)
plt.savefig("./figures/cytotoxicity.tiff", format="tiff")
# plt.show()

### plot corner - growth ################################################################################################
### mean, median, 95% credible regions ###
df = pd.DataFrame(np.log10(chains))

mean = []
median = []
lower = []
upper = []
for ii in range(ndim):
    mean.append(np.mean(chains[:,ii]))
    median.append(np.median(chains[:, ii]))
    lower.append(np.percentile(chains[:, ii],2.5))
    upper.append(np.percentile(chains[:, ii],97.5))

par_names = "A_max", "CC_50", "N_A"
with open('parameters_cc50.txt', 'w') as f:
    f.writelines(['parameter','\t','maxlik','\t','mean','\t','median','\t','95 lower','\t','95 upper','\n'])
    for jj in range(ndim):
        f.writelines([par_names[jj],'\t',
                      str(maxlik_parameters[jj]),'\t',
                      str(mean[jj]),'\t',
                      str(median[jj]),'\t',
                      str(lower[jj]),'\t',
                      str(upper[jj])])
        f.write('\n')

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

labels = r"log$_{10}$ $A_\mathrm{max}$", r"log$_{10}$ $\mathrm{CC}_{50}$", r"log$_{10}$ $N_\mathrm{A}$"

fig, axs = plt.subplots(ndim, ndim)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(ndim):
    for j in range(ndim):
        if i < j:
            axs[i,j].axis("off")
        elif i == j:
            sns.kdeplot(data=np.log10(chains[:, i]),
                        ax=axs[i, j],
                        color="red",
                        fill=True)
            if i < ndim-1:
                axs[i, j].set_xticks([])
                axs[i, j].set_xlabel('')
                axs[i, j].set_xticklabels([])
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel('')
            axs[i, j].set_yticklabels([])
        else:
            corrplt = sns.kdeplot(data=df, x=df[j], y=df[i], ax=axs[i, j], color="red", linewidths=0.5)
            if i != ndim - 1:
                corrplt.set(ylabel=None)
            if i != ndim - 1:
                corrplt.set(xlabel=None)
            if j  != 0:
                corrplt.set(ylabel=None)
            axs[i, j].scatter(np.log10(maxlik_parameters[j]), np.log10(maxlik_parameters[i]),
                              s=24,
                              color="black",
                              alpha=1)
            axs[i, j].axvline(x = np.log10(maxlik_parameters[j]),
                              color="black",
                              linestyle='-',
                              linewidth=0.5,
                              alpha=1)
            axs[i, j].axhline(y = np.log10(maxlik_parameters[i]),
                              color="black",
                              linestyle='-',
                              linewidth=0.5,
                              alpha=1)
            correl, _ = pearsonr(np.log10(chains[:,j]), np.log10(chains[:,i]))
            axs[i,j].text(0.75, 0.85, round(correl, 2),
                          fontsize=MEDIUM_SIZE,
                          transform=axs[i,j].transAxes,
                          color='black',
                          weight='bold')
            if i < ndim - 1:
                axs[i, j].set_xticklabels([])
            if j > 0:
                axs[i, j].set_yticklabels([])
    axs[ndim - 1, i].set_xlabel(labels[i])
    axs[ndim - 1, i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if i == 0:
        axs[i, 0].set_yticklabels([])
    else:
        axs[i, 0].set_ylabel(labels[i], rotation=90, labelpad=5)

# print(axs[0,0].get_position())
# axs[0,0].text(0.45, 5, "(B)",
#                fontsize=BIGG_SIZE,
#                color='black')
fig.tight_layout()

plt.savefig("../LaTeX/figures/corner_cytotoxicity.pdf", format="pdf", transparent=True)
plt.savefig("./figures/corner_cytotoxicity.tiff", format="tiff")
plt.show()