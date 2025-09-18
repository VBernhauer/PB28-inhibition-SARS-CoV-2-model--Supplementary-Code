#! /usr/bin/env python
from __future__ import division
from pickle import dump, load
from scipy.stats import pearsonr
from scipy.optimize import least_squares
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import emcee

from mcmc import IC50_lower, parameters_init

chains = "./chains_IC50"
if not os.path.exists(chains):
    os.makedirs(chains)

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

### inhibition PB28 at 72h #############################################################################################
### read data ###
file        = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="virus inhibition PB28 at 72h")
C           = file["PB28 [uM]"].values[:]
data        = np.log10(file.values.T[1:4])
std_data    = np.std(data,axis=0,ddof=1)

########################################################################################################################
def lognormpdf(data, model, stdev):
    return - 0.5 * ((data - model) / stdev) ** 2 - np.log(np.sqrt(2 * np.pi) * stdev)


### log-probability ###
def logprob(parameters):
    lp = logprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(parameters)

### priors ###
def logprior(parameters):
    if (not all(x >= bound_lower[id_lower] for id_lower, x in enumerate(parameters)) or
        not all(x < bound_upper[id_upper] for id_upper, x in enumerate(parameters))):
        return -np.inf
    return 0.0


def loglike(parameters):
    V = virus(parameters, C)
    value = 0
    for ii in range(np.size(data,0)):
        value = value + np.sum(lognormpdf(data[ii],V,std_data[ii]))
    return value

def virus(parameters, c):
    V = parameters[0] + (parameters[1] - parameters[0]) / (1 + 10 ** (parameters[3] * (np.log10(c) - np.log10(parameters[2]))))
    logV = np.log10(V)
    return logV

def residuals_virus(parameters, c, data):
    res = data - virus(parameters, c)
    return res


V0_min_lower  = 0
V0_min_upper  = np.inf
V0_max_lower  = 0
V0_max_upper  = np.inf
IC_50_lower   = 0
IC_50_upper   = np.inf
N_lower       = 0
N_upper       = np.inf

bound_lower = [V0_min_lower,
               V0_max_lower,
               IC_50_lower,
               N_lower]
bound_upper = [V0_min_upper,
               V0_max_upper,
               IC_50_upper,
               N_upper]

V0_min      = np.log10(1e+4)
V0_max      = np.log10(1e+6)
IC_50       = np.log10(1e-1)
N           = np.log10(1e+0)

parameters = V0_min, V0_max, IC_50, N

#### MCMC #############################################################################################################
pos = 10 ** (parameters + 1e-4 * np.random.randn(2 * len(parameters), len(parameters)))
nwalkers, ndim = np.shape(pos)

sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
Nsamples = 20000  # number of final posterior samples

print("MCMC...")
sampler.run_mcmc(pos, Nsamples, progress=True)

chains = sampler.get_chain(discard=int(Nsamples/2), thin=5, flat=True)
log_prob = sampler.get_log_prob(discard=int(Nsamples/2), thin=5, flat=True)

with open('chains_IC50/logprob.obj', 'wb') as a:
    dump(log_prob, a)
with open('chains_IC50/chains.obj', 'wb') as a:
    dump(chains, a)

########################################################################################################################
with open('chains_IC50/chains.obj', 'rb') as a:
    chains = load(a)

with open('chains_IC50/logprob.obj', 'rb') as b:
    log_prob = load(b)

labels = (r"log$_{10}$ $\mathrm{V}_\mathrm{{0,min}}$", r"log$_{10}$ $\mathrm{V}_\mathrm{{0,max}}$",
          r"log$_{10}$ $\mathrm{IC}_{50,\epsilon}$", r"log$_{10}$ $N_{\epsilon}$")
ndim = len(labels)

cc = np.concatenate([np.linspace(0.01, 0.99, 99), np.linspace(1, 10, 91)])

### index of the maximum likelihood value ##############################################################################
idx_max_likelihood = np.argmax(log_prob)
maxlik_parameters = chains[idx_max_likelihood,:]
maxlik_solution = virus(chains[idx_max_likelihood,:], cc)

# for i in range(len(cc)):
#     print([10**maxlik_solution[i],cc[i]])

### get the bounds on the solutions ####################################################################################
solution = []
for i in range(int(np.size(chains)/ndim)):
    solution.append(virus(chains[i,:], cc))
solution = np.array(solution)

solution_min = []
solution_max = []
for idx, concentration in enumerate(cc):
    solution_min_idx = np.percentile(solution[:,idx], 2.5)
    solution_min.append(solution_min_idx)
    solution_max_idx = np.percentile(solution[:,idx], 97.5)
    solution_max.append(solution_max_idx)

    # solution_min_idx = np.min(solution[:,idx])
    # solution_min.append(solution_min_idx)
    # solution_max_idx = np.max(solution[:,idx])
    # solution_max.append(solution_max_idx)

### fit Hill to viral loads
guess = np.array([10 ** maxlik_solution[-1], 10 ** maxlik_solution[0], 0.5, 2])
opt_parameters = least_squares(residuals_virus, guess, args=(cc, maxlik_solution))
IC50_parameters = opt_parameters.x

########################################################################################################################
### mean, median, 95% credible regions #################################################################################
mean = []
median = []
lower = []
upper = []
for ii in range(ndim):
    mean.append(np.mean(chains[:, ii]))
    median.append(np.median(chains[:, ii]))
    lower.append(np.percentile(chains[:, ii], 2.5))
    upper.append(np.percentile(chains[:, ii], 97.5))

par_names = "V0_min", "V0_max", "IC_50", "n_epsilon"
with open('parameters_ic50.txt', 'w') as f:
    f.writelines(
        ['parameter', '\t', 'maxlik', '\t', 'mean', '\t', 'median', '\t', '95 lower', '\t', '95 upper', '\n'])
    for jj in range(ndim):
        f.writelines([par_names[jj], '\t',
                      str(maxlik_parameters[jj]), '\t',
                      str(mean[jj]), '\t',
                      str(median[jj]), '\t',
                      str(lower[jj]), '\t',
                      str(upper[jj])])
        f.write('\n')

### plot ###############################################################################################################
fig = plt.figure(figsize=(5, 4))

markersize = 8
capsize = 8
alpha = 0.35
alpha_data = 1
linewidth = 1.5
elinewidth = 1.5
fontsize = 12

plt.subplot(1, 1, 1)
frame = fig.gca()
frame.axes.xaxis.set_ticks(np.log10(C))
frame.axes.xaxis.set_ticklabels(["0.01", "0.1", "0.2", "0.5", "2", "10"], fontsize=fontsize)
plt.xlim(-2.1, 1.1)
frame.axes.yaxis.set_ticks([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
frame.axes.yaxis.set_ticklabels(["3.5", "4", "4.5", "5", "5.5", "6", "6.5"], fontsize=fontsize)
plt.ylim(3.3, 6.7)
plt.xlabel(r"PB28 concentration (\textmu M)", fontsize=fontsize)
plt.ylabel(r"Viral load (log$_{10}$ PFU$_\mathrm{e}$/mL", fontsize=fontsize)
# plt.fill_between(np.log10(cc), solution_min, solution_max, facecolor="grey", edgecolor="grey", alpha=0.3)
plt.plot(np.log10(cc), maxlik_solution, color="green",
                                                 linestyle="-",
                                                 linewidth=1,
                                                 alpha=1)
plt.plot(np.log10(cc), solution_min, color="green",
                                                 linestyle="--",
                                                 linewidth=1,
                                                 alpha=0.5)
plt.plot(np.log10(cc), solution_max, color="green",
                                                 linestyle="--",
                                                 linewidth=1,
                                                 alpha=0.5)
plt.plot(np.log10(IC50_parameters[2]) * np.ones(10),
                 np.linspace(3.3, virus(IC50_parameters, IC50_parameters[2]), 10),
                 color="black",
                 linestyle=":",
                 alpha=0.5)
plt.plot(np.linspace(-2.1, np.log10(IC50_parameters[2]), 10),
                 np.ones(10) * virus(IC50_parameters, IC50_parameters[2]),
                 color="black",
                 linestyle=":",
                 alpha=0.5)
for line in data:
    plt.plot(np.log10(C), line, marker="o",
                                        color="green",
                                        markeredgecolor="black",
                                        linestyle=" ",
                                        markersize=markersize,
                                        alpha=alpha_data)
plt.text(-2, 3.5, "IC$_{50}$ = " + str(round(maxlik_parameters[2], 3)) + r" \textmu M",
               fontsize=fontsize,
               color='green')#,
               # weight='bold')
fig.tight_layout()
plt.savefig("../LaTeX/figures/inhibition.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/inhibition.tiff", format="tiff")


### corner plot ########################################################################################################
n_bins = 50

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

df = pd.DataFrame(np.log10(chains))

fig, axs = plt.subplots(ndim, ndim, figsize=(7, 5))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(ndim):
    for j in range(ndim):
        if i < j:
            axs[i, j].axis("off")
        elif i == j:
            sns.kdeplot(data=np.log10(chains[:, i]),
                        ax=axs[i, j],
                        color="green",
                        fill=True)
            # axs[i, j].hist(np.log10(chains[:, i]),
            #                bins=n_bins,
            #                color=(0, 0, 0, 0.1),
            #                linewidth=0.5,
            #                edgecolor="green")
            if i < ndim-1:
                axs[i, j].set_xticks([])
                axs[i, j].set_xlabel('')
                axs[i, j].set_xticklabels([])
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel('')
            axs[i, j].set_yticklabels([])
        else:
            corrplt = sns.kdeplot(data=df, x=df[j], y=df[i], ax=axs[i, j], color="green", linewidths=0.5)
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
            axs[i,j].text(0.7, 0.8, round(correl, 2),
                          fontsize=MEDIUM_SIZE,
                          transform=axs[i,j].transAxes,
                          color='black',
                          weight='bold')
            if i < ndim - 1:
                axs[i, j].set_xticklabels([])
            if j > 0:
                axs[i, j].set_yticklabels([])
        axs[ndim - 1, i].set_xlabel(labels[i])
        if i < ndim - 1:
            axs[ndim - 1, i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if 0 < i < ndim - 1:
            axs[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if i == 0:
            axs[i, 0].set_yticklabels([])
        else:
            axs[i, 0].set_ylabel(labels[i], rotation=90, labelpad=5)

plt.tight_layout()
plt.savefig("../LaTeX/figures/inhibition_corner.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/inhibition_corner.tiff", format="tiff")
plt.show()