#! /usr/bin/env python

from pickle import load
from scipy.stats import mannwhitneyu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

with open('chains/chains.obj', 'rb') as a:
    chains = load(a)

with open('chains_IC50/chains.obj', 'rb') as b:
    chains_IC50 = load(b)


mw_IC50_stat, mw_IC50_pvalue    = mannwhitneyu(chains[:, 5], chains_IC50[:, 2])
print('MW test for IC50: ', mw_IC50_pvalue)
mw_N_stat, mw_N_pvalue          = mannwhitneyu(chains[:, 6], chains_IC50[:, 3])
print('MW test for N: ', mw_N_pvalue)

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

sns.kdeplot(data=chains[:, 5],
            ax=axs[0],
            color="blue",
            fill=True)
sns.kdeplot(data=np.log10(chains_IC50[:, 2]),
            ax=axs[0],
            color="green",
            fill=True)
sns.kdeplot(data=chains[:, 6],
            ax=axs[1],
            color="blue",
            fill=True)
sns.kdeplot(data=np.log10(chains_IC50[:, 3]),
            ax=axs[1],
            color="green",
            fill=True)

axs[0].set_ylim([0, 10])
axs[1].set_ylim([0, 10])
axs[0].set_yticks([0,2,4,6,8,10])
axs[1].set_yticks([0,2,4,6,8,10])
axs[0].set_xlim([-2, 0])
axs[1].set_xlim([-0.5, 1])
axs[1].set_xticks([-0.5,0,0.5,1])
axs[0].set_xlabel("Log$_{10}$ value")
axs[1].set_xlabel("Log$_{10}$ value")

mp = [[], []]
mp[0] = mpatches.Patch(color="blue", linewidth=0)
mp[1] = mpatches.Patch(color="green", linewidth=0)
axs[0].legend(mp, [r"time-resolved, ${IC}_{50,\epsilon}$", r"end-point, ${IC}_{50}$"], loc='upper left')
axs[1].legend(mp, [r"time-resolved, $N_{\epsilon}$", r"end-point, $N$"], loc='upper right')

plt.tight_layout()
plt.savefig("../LaTeX/figures/pk_posteriors.pdf",
            format="pdf", transparent=True)
plt.savefig("./figures/pk_posteriors.tiff", format="tiff")
plt.show()