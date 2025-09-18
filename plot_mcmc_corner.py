#! /usr/bin/env python

from pickle import load
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import os

### font ###############################################################################################################
# print(matplotlib.rcParams['font.family'])
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'Helvetica'
# print(matplotlib.rcParams['font.family'])

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
########################################################################################################################
fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

with open('chains/chains.obj', 'rb') as a:
    chains = load(a)

with open('chains/logprob.obj', 'rb') as c:
    log_prob = load(c)

labels = (r"log$_{10}$ $\beta$",
          r"log$_{10}$ $\tau_\mathrm{E}$",
          r"log$_{10}$ $\tau_\mathrm{I}$",
          r"log$_{10}$ $p$",
          r"log$_{10}$ $\omega_0$",
          r"log$_{10}$ ${IC}_{50}$",
          r"log$_{10}$ $N_{\epsilon}$",
          r"$n_\mathrm{E}$")
ndim = len(labels)

### index of the maximum likelihood value ###
idx_max_likelihood              = np.argmax(log_prob)
max_likelihood_parameters       = chains[idx_max_likelihood,:]
max_likelihood_parameters[-1]   = int(max_likelihood_parameters[-1])

### mean, median, 95% credible regions ###
mean = []
median = []
lower = []
upper = []
for ii in range(ndim):
    mean.append(np.mean(chains[:,ii]))
    median.append(np.median(chains[:, ii]))
    lower.append(np.percentile(chains[:, ii],2.5))
    upper.append(np.percentile(chains[:, ii],97.5))


par_names = "beta", "tau_L", "tau_I", "p", "omega_0", "IC_50", "N_eps", "n_L"
with open('parameters.txt', 'w') as f:
    f.writelines(['parameter','\t','maxlik','\t','mean','\t','median','\t','95 lower','\t','95 upper','\n'])
    for jj in range(ndim):
        if jj != ndim - 1:
            f.writelines([par_names[jj],'\t',
                          str(10**max_likelihood_parameters[jj]),'\t',
                          str(10**mean[jj]),'\t',
                          str(10**median[jj]),'\t',
                          str(10**lower[jj]),'\t',
                          str(10**upper[jj])])
            f.write('\n')
        else:
            f.writelines([par_names[jj],'\t',
                          str(int(max_likelihood_parameters[jj])),'\t',
                          str(int(mean[jj])),'\t',
                          str(int(median[jj])),'\t',
                          str(int(lower[jj])),'\t',
                          str(int(upper[jj]))])
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

df = pd.DataFrame(chains)
n_bins = 50

fig, axs = plt.subplots(ndim, ndim, figsize=(15, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(ndim):
    for j in range(ndim):
        if i < j:
            axs[i, j].axis("off")
        elif i == j:
            # if i == ndim-1:
                # axs[i,j].hist(chains[:,i].astype(int),
                #                    bins=n_bins,
                #                    color=(0, 0, 0, 0.1),
                #                    linewidth=0.5,
                #                    edgecolor="grey")
                # his = np.histogram(chains[:, i].astype(int))
                # offset = 0.4
                # axs[i,j].bar(his[1][1:],his[0],
                #           color=(0, 0, 0, 0.1),
                #           linewidth=1,
                #           edgecolor="grey")
            # else:
            # axs[i, j].hist(chains[:, i],
            #                    bins=n_bins,
            #                    color=(0, 0, 0, 0.1),
            #                    linewidth=0.5,
            #                    edgecolor="blue")
            sns.kdeplot(data=chains[:, i],
                        ax=axs[i, j],
                        color="blue",
                        fill=True)
            if i < ndim-1:
                axs[i, j].set_xticks([])
                axs[i, j].set_xlabel('')
                axs[i, j].set_xticklabels([])
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel('')
            axs[i, j].set_yticklabels([])
        else:
            corrplt = sns.kdeplot(data=df, x=df[j], y=df[i], ax=axs[i, j], color="blue", linewidths=0.5)
            if i != ndim - 1:
                corrplt.set(ylabel=None)
            if i != ndim - 1:
                corrplt.set(xlabel=None)
            if j  != 0:
                corrplt.set(ylabel=None)
            axs[i, j].scatter(max_likelihood_parameters[j], max_likelihood_parameters[i],
                              s=24,
                              color="black",
                              alpha=1)
            axs[i, j].axvline(x = max_likelihood_parameters[j],
                              color="black",
                              linestyle='-',
                              linewidth=0.5,
                              alpha=1)
            axs[i, j].axhline(y = max_likelihood_parameters[i],
                              color="black",
                              linestyle='-',
                              linewidth=0.5,
                              alpha=1)
            correl, _ = pearsonr(chains[:,j], chains[:,i])
            axs[i,j].text(0.75, 0.8, round(correl, 2),
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
axs[ndim - 1, ndim - 1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axs[ndim - 1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.savefig("../LaTeX/figures/corner.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/corner.tiff", format="tiff")
plt.show()