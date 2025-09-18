#! /usr/bin/env python
from __future__ import division


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig_dir = "./figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

### inhibition PB28 at 72h #############################################################################################
### read data ###
file_inhibition = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="virus inhibition PB28 at 72h")
c_file_inhibition = file_inhibition["PB28 [uM]"].values[:]
data_file_inhibition = file_inhibition.values.T[0:6]

### viral load PB28 0uM ################################################################################################
### read data ###
file_virus_0 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 0uM")
t_file_virus = file_virus_0["Time"].values[:]
data_file_virus_0 = file_virus_0.values.T[1:7]

### viral load PB28 0.5uM ##############################################################################################
### read data ###
file_virus_05 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 0.5uM")
data_file_virus_05 = file_virus_05.values.T[1:7]

### viral load PB28 0.5uM #########################################################################################
### read data ###
file_virus_5 = pd.read_excel(io="./supplementary_data.xlsx", sheet_name="viral load 5uM")
data_file_virus_5 = file_virus_5.values.T[1:7]

### plot ###
fontsize = 12
markersize = 6
alpha = 1

colors = ["blueviolet",
          "blue",
          "cyan"]

fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))

axs[0].xaxis.set_ticks(np.log10(c_file_inhibition))
axs[0].xaxis.set_ticklabels(["0.01", "0.1", "0.2", "0.5", "2", "10"],fontsize=fontsize)
axs[0].yaxis.set_ticks([3.5, 4.0, 4.5, 5., 5.5, 6., 6.5])
axs[0].yaxis.set_ticklabels([3.5, 4.0, 4.5, 5., 5.5, 6., 6.5],fontsize=fontsize)
axs[0].set_xlim(-2.1, 1.1)
axs[0].set_ylim([3.5, 6.5])
axs[0].set_xlabel(r"PB28 concentration (\textmu M)", fontsize=fontsize),
axs[0].set_ylabel(r"Viral load (log$_{10}$ PFU$_\mathrm{e}$/mL)", fontsize=fontsize)
axs[0].text(-2.0, 6.2, "(A) end-point infection",
               fontsize=fontsize,
               color='black')
for line_c_file_inhibition in data_file_inhibition:
    axs[0].plot(np.log10(c_file_inhibition), np.log10(line_c_file_inhibition), marker="o",
                                                                                  color="green",
                                                                                  markeredgecolor="black",
                                                                                  linestyle=" ",
                                                                                  markersize=markersize,
                                                                                  alpha=alpha)

axs[1].xaxis.set_ticks(t_file_virus)
axs[1].xaxis.set_ticklabels(t_file_virus,fontsize=fontsize)
axs[1].yaxis.set_ticks([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
axs[1].yaxis.set_ticklabels([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7],fontsize=fontsize)
axs[1].set_xlim(14.0, 100.0)
axs[1].set_ylim(3, 7)
axs[1].set_xlabel("Time (hours post-infection)", fontsize=fontsize),
axs[1].set_ylabel(r"Viral load (log$_{10}$ PFU$_\mathrm{e}$/mL)", fontsize=fontsize)
axs[1].text(17, 6.6, "(B) time-resolved infection",
               fontsize=fontsize,
               color='black')
for line_data_file_virus_0 in data_file_virus_0:
    axs[1].plot(t_file_virus, np.log10(line_data_file_virus_0), marker="o",
                                                                   color=colors[0],
                                                                   markeredgecolor="black",
                                                                   linestyle=" ",
                                                                   markersize=markersize,
                                                                   alpha=alpha)
for line_data_file_virus_05 in data_file_virus_05:
    axs[1].plot(t_file_virus, np.log10(line_data_file_virus_05), marker="o",
                                                                   color=colors[1],
                                                                   markeredgecolor="black",
                                                                   linestyle=" ",
                                                                   markersize=markersize,
                                                                   alpha=alpha)
for line_data_file_virus_5 in data_file_virus_5:
    axs[1].plot(t_file_virus, np.log10(line_data_file_virus_5), marker="o",
                                                                   color=colors[2],
                                                                   markeredgecolor="black",
                                                                   linestyle=" ",
                                                                   markersize=markersize,
                                                                   alpha=alpha)

point_grey = plt.Line2D([0], [0],
                    marker='o',
                    markersize=markersize,
                    markeredgecolor='black',
                    markerfacecolor=colors[0],
                    linestyle='',
                    alpha=alpha)
point_blue = plt.Line2D([0], [0],
                    marker='o',
                    markersize=markersize,
                    markeredgecolor='black',
                    markerfacecolor=colors[1],
                    linestyle='',
                    alpha=alpha)
point_cyan = plt.Line2D([0], [0],
                    marker='o',
                    markersize=markersize,
                    markeredgecolor='black',
                    markerfacecolor=colors[2],
                    linestyle='',
                    alpha=alpha)
axs[1].legend([point_grey, point_blue, point_cyan],
              [r'control', r'PB28 0.5 \textmu M', r'PB28 5 \textmu M'],
              loc="lower right")

fig.tight_layout()
plt.savefig("../LaTeX/figures/data.pdf",
            format="pdf", transparent=True)
plt.savefig("figures/data.tiff", format="tiff")
plt.show()