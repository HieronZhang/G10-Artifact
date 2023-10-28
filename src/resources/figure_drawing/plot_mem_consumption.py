import enum
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import pickle
import sys
from glob import glob
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator
import matplotlib.backends.backend_pdf
import matplotlib
from networkx.drawing.nx_agraph import to_agraph 
import os
import pandas as pd
from PyPDF2 import PdfMerger
import glob
import math


TEXT_ONLY = False

plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5


line_styles = ['--', '-', '-.', ':']
hatches = ["", "\\", "//", "||"]
colors = ['#ff796c', 'plum', '#95d0fc', 'gray']
line_colors = ['brown', 'forestgreen', '#23a8eb', 'gray', 'black']
markers = ['.', '.', '*', 'v', '^']
    

    
def plot_timeline(ax: plt.Axes, results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False, scaling=1.0, markevery=1, legend=False):
    if TEXT_ONLY:
        return
    
    # ax.figure(figsize=(8, 3), dpi=300)
    max_y_value = - float('inf')
    # values = np.arange()
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    if step:
        plot_func = ax.step
    else:
        plot_func = ax.plot
    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        seriess = np.array(series) * scaling
        max_y_value = max(max_y_value, max(seriess))
    if aggregate:
        agg_seriess = np.sum(series for _, series in results.items())
        agg_seriess = np.array(agg_seriess) * scaling
        max_y_value = max(max_y_value, max(agg_seriess))

    import pandas as pd

    results["active"] = pd.Series(results["active"]).rolling(8).max().dropna().tolist()
    results["all"] = pd.Series(results["all"]).rolling(1).max().dropna().tolist()

    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        series = np.array(series) * scaling / max_y_value
        plot_func(np.arange(len(series)), series, label=plot_policy, color=line_colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=3, markevery=markevery)
    if aggregate:
        agg_series = np.sum(series for _, series in results.items())
        agg_series = np.array(agg_series) * scaling / max_y_value
        plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
        
    if legend:
        ax.legend(ncol=4, fontsize=16, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.35))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    ax.set_xlim(0, len(results["active"]))

    # plt.xlim([0, TIMESTEPS])
    if max_y_value != 0 and max_y_value != - float('inf'):
        ax.set_ylim(ax.get_ylim()[0], 1.2*max_y_value / max_y_value )
        #plt.locator_params(axis='y', nbins=5)
        # num_yticks = 5
        # # nearest_unit = 10**math.floor(math.log10(max_y_value // num_yticks))
        # ytick_gap = max_y_value*1.2 / num_yticks
        # plt.yticks(np.arange(num_yticks) * ytick_gap)
    ax.grid(which='major', axis='y', color='#000000', linestyle='--')
    # ax.tight_layout()
    # ax.set_yticks([i for i in ax.get_yticks()])
    ax.set_yticklabels([f"{i:.1}%" if i < 1 else f"{int(i):d}%" for i in [i * 100 for i in ax.get_yticks()]])
    # ax.savefig(f"{filename}")
    # ax.clf()

def plot_multi_timeline(multi_results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False):
    if TEXT_ONLY:
        return
    
    num_subplots = len(multi_results)
    if num_subplots == 0:
        return
    fig = plt.figure(figsize=(10, 4*num_subplots), dpi=300)
    axes = fig.subplots(nrows=num_subplots, ncols=1)
    max_y_value = - float('inf')
    for i, (top_label, results) in enumerate(multi_results.items()):
        if num_subplots == 1:
            ax = axes
        else:
            ax = axes[i]
        if step:
            plot_func = ax.step
        else:
            plot_func = ax.plot
        for i, (second_label, series) in enumerate(results.items()):
            if cumulative:
                series = np.cumsum(series)
            plot_func(np.arange(len(series)), series, label=second_label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)])
            max_y_value = max(max_y_value, max(series))
        if aggregate:
            agg_series = np.sum(series for _, series in results.items())
            plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
            max_y_value = max(max_y_value, max(agg_series))
        ax.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(top_label)
        ax.set_xlim([0, TIMESTEPS])
        if max_y_value != 0 and max_y_value != - float('inf'):
            ax.set_ylim([0, 1.2*max_y_value])
        ax.grid(b=True, which='major', axis='y', color='#000000', linestyle='--')
    
    plt.tight_layout()  
    plt.savefig(f"{filename}")
    plt.clf()



from fig_common import *

batch_sizes = [4, 8, 16, 32, 64, 128, 256]
model_names = ["OPT_1.3", "OPT_2.7", "OPT_6.7", "OPT_13"]

for batch_size in batch_sizes:
    for model_name in model_names:
        
        try: 
            exec(open(f'../../../results/{model_name}/{batch_size}-lru_NNMemConsumptionLog.py').read())

            title = f"{model_name}_bs{batch_size}_mem_consumption"
            Figure = plt.figure(figsize=(8, 10))
            PDF = PdfPages("output/" + title + ".pdf")

        
            live = active
            real = total
            motiv1 = {"all" : real, "active" : live}
            ax = Figure.add_subplot(211)
            plot_timeline(ax, motiv1, "mem_consumption_opt", f"CUDA Kernel Index\n{model_name} BS{batch_size}", " ", markevery=1, legend=True)

            Figure.text(-0.15, 0.5, "Memory Consumption", rotation=90, \
                horizontalalignment='center', verticalalignment='center', \
                transform=ax.transAxes)

            Figure.tight_layout(pad=1.)

            PDF.savefig(Figure, bbox_inches='tight')
            PDF.close()

        except:
            print(f"{model_name} batch {batch_size} not found.")