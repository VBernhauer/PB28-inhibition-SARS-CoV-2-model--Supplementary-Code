#! /usr/bin/env python
from pickle import load
from scipy.stats import mannwhitneyu

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

with open('chains/logprob.obj', 'rb') as a:
    logprob_p = load(a)

with open('Reduced infection rate/chains/logprob.obj', 'rb') as b:
    logprob_beta = load(b)


mw_stat, mw_pvalue    = mannwhitneyu(logprob_p, logprob_beta)
print('MW p-value for model differences: ', mw_pvalue)

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

sns.kdeplot(data=logprob_p,
            ax=axs,
            color="gray",
            fill=True)
sns.kdeplot(data=logprob_beta,
            ax=axs,
            color="orange",
            fill=True)

axs.set_xlim([15, 40])
axs.set_ylim([0, 0.3])
axs.set_xlabel("Log-likelihood value")

mp = [[], []]
mp[0] = mpatches.Patch(color="gray", linewidth=0)
mp[1] = mpatches.Patch(color="orange", linewidth=0)
axs.legend(mp, [r"PB28-induced reduction of $p$", r"PB28-induced reduction of $\beta$"], loc='upper left')

plt.tight_layout()
plt.savefig("./figures/FigS15.pdf", format="pdf", transparent=True)
plt.savefig("./figures/FigS15.tiff", format="tiff")
plt.show()