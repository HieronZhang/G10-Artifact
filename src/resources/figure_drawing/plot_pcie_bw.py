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
line_colors = ["#669ABA", "#be1420", 'gray', '#23a8eb', 'black']
markers = ['.', '.', '*', 'v', '^']
    

def plot_timeline(ax: plt.Axes, results, filename, xlabel="Hours", ylabel="Migrate Overhead (hrs)", cumulative=False, step=False, aggregate=False, scaling=1.0, markevery=1, legend=False, yscale_log=False):
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

    for label in ["active", "input", "weight", "intermediate"]:
        if label in results.keys():
            results[label] = pd.Series(results[label]).rolling(6).max().dropna().tolist()
            print(label, max_y_value, max(results[label]))
    if "all" in results:
        results["all"] = pd.Series(results["all"]).rolling(6).max().dropna().tolist()

    # line_styles = ["-", "-", "-", "-"]
    line_styles = ["-", "--", "-.", ":"]
    for i, (plot_policy, series) in enumerate(results.items()):
        if cumulative:
            series = np.cumsum(series)
        series = np.array(series) * scaling 
        plot_func(np.arange(len(series)), series, label=plot_policy, color=line_colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=3, markevery=markevery)
    if aggregate:
        agg_series = np.sum(series for _, series in results.items())
        agg_series = np.array(agg_series) * scaling 
        plot_func(np.arange(len(agg_series)), agg_series, label="total", color="purple", linewidth=2)
        
    if legend:
        ax.legend(ncol=4, fontsize=18, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.4))
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if yscale_log:
        ax.set_yscale('log')
    else:
        ax.set_yticks([0, 10, 20, 30])

    ax.set_xlim(0, max(len(s) for s in results.values()))

    ax.set_yticklabels([f'{int(i)}' for i in ax.get_yticks()])

    # plt.xlim([0, TIMESTEPS])
    if max_y_value != 0 and max_y_value != - float('inf'):
        ax.set_ylim(ax.get_ylim()[0], 1.15*max_y_value )
        #plt.locator_params(axis='y', nbins=5)
        # num_yticks = 5
        # # nearest_unit = 10**math.floor(math.log10(max_y_value // num_yticks))
        # ytick_gap = max_y_value*1.2 / num_yticks
        # plt.yticks(np.arange(num_yticks) * ytick_gap)
    # ax.set_ylim(0.002, 1.2)
    ax.grid(which='major', axis='y', color='#000000', linestyle='--')
    # ax.tight_layout()
    # print(ax.get_yticks(), [f'{i:0.0%}' for i in ax.get_yticks()])
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
        ax.legend(ncol=4, fontsize=12, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.4))
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

title = "bw_untilization_granite"
Figure = plt.figure(figsize=(16, 5))
PDF = PdfPages("output/" + title + ".pdf")

AFTER_LOCAL_PASS = 0
TOTAL_LIVE = 1
BEFORE_LOCAL_PASS = 2
# GLOBAL = 3

selection = 1

directory = '../../../results/bw_profiling_test/'

# exec(open(directory + 'granite-rank0-pci8/statistics/pre_dealloc.py').read())
# live = active
# live_breakdown = active_breakdown
# live_input = [item[0] for item in live_breakdown]
# live_weight = [item[1] + 1 for item in live_breakdown]
# live_intermediate = [item[2] for item in live_breakdown]
# real = total
# global_input = [input_size for _ in real]
# global_weight = [global_weight for _ in real]
# global_intermediate = [s - global_input[0] - global_weight[0] for s in real]



rank0_out_bw_list = [15.689782000000001, 22.174052000000003, 23.717622, 23.248488000000002, 21.502585000000003, 25.01309, 25.377412, 25.181717999999996, 21.768304, 1.842162, 3.336921, 0.0, 0.0, 0.857594, 0.613846, 0.669927, 8.593452, 25.717188999999998, 26.327462, 22.223253, 20.732336, 10.150309, 1.782453, 0.617834, 0.251735, 0.0, 0.297709, 0.0, 3.065072, 25.416617000000002, 27.242202, 25.054256, 20.964354, 11.175056, 3.483241, 0.492889, 0.283475, 0.277206, 0.494265, 0.292272, 0.28868, 26.031754999999997, 25.674294000000003, 24.279925, 20.977942, 12.436665000000001, 3.205399, 1.226112, 0.236027, 0.0, 0.28703, 0.23446, 0.57794, 23.520716, 24.431517999999997, 26.487277, 22.411786, 15.076346000000001, 5.836928, 1.029008, 0.0, 0.487404, 0.556352, 0.306497, 0.243557, 18.376341, 26.05818, 25.838959, 22.216635, 17.135674, 6.217725, 0.555992, 0.244062, 0.589173, 0.531416, 0.248901, 0.869927, 11.564494, 24.959784, 26.482028, 21.904569000000002, 20.529214, 9.159276, 0.29287, 0.583943, 0.445034, 0.25307, 0.0, 0.472341, 7.157722, 26.030176, 27.794407, 24.868347999999997, 20.79021, 9.571437, 3.058602, 0.525142, 0.306467, 0.41792, 0.554708, 0.615631, 0.9418110000000001, 25.559558, 24.850534, 23.129739, 20.476227, 12.785291, 3.729771, 0.277841, 0.500776, 0.281644, 0.248479, 0.252959, 0.28757, 24.020295, 27.276493000000002, 27.644468, 21.640717000000002, 15.427122, 5.672114, 0.588671, 0.92263, 0.23364, 0.294611, 0.618383, 0.0, 21.791805, 23.275845, 22.056221999999998, 20.400488000000003, 16.377556, 7.843725, 0.470758, 0.0, 0.607999, 0.0, 0.288449, 0.613918, 13.412946, 27.647396, 27.603505, 25.557093000000002, 18.691774000000002, 5.847597, 0.943008, 0.252677, 1.032096, 0.622523, 0.0, 0.260816, 9.057277, 23.806815, 21.184234, 22.204189, 21.002651999999998, 10.061897, 2.213702, 0.0, 0.273896, 0.0, 0.858127, 0.541865, 0.9685739999999999, 25.343764, 27.938765, 28.023209, 21.984398, 12.143887, 1.765477, 0.23559, 0.849678, 0.306934, 0.944056, 0.59322, 0.616308, 22.373643, 22.064032, 23.109971, 23.652537000000002, 9.707578999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.432037, 2.708338, 4.49162, 4.38819, 3.16453, 4.629843, 4.073187, 3.834936, 4.437606, 3.3301239999999996, 4.6635349999999995, 4.0927240000000005, 3.8009910000000002]
rank0_in_bw_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.971382, 0.0, 0.0, 0.0, 2.40139, 16.406531, 14.67036, 12.829915, 15.222290999999998, 6.752308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.666224, 20.116258000000002, 16.372604, 15.355862000000002, 17.612651, 16.67172, 3.324697, 0.383134, 0.0, 0.0, 0.0, 0.0, 18.30807, 18.577287, 19.222656999999998, 15.307666000000001, 14.137530000000002, 17.299288, 0.292272, 2.020757, 0.0, 0.0, 0.0, 0.0, 16.812527000000003, 18.431042, 17.165565, 16.52186, 15.795489, 15.499628999999999, 6.330429, 0.866911, 0.0, 0.0, 0.0, 0.0, 11.177291, 22.840153, 16.978626, 15.620698, 17.546554999999998, 16.690568, 8.888404, 0.73067, 0.0, 0.0, 0.0, 0.0, 9.38941, 20.914165, 13.065804, 20.501241, 15.613075, 14.082511, 12.445034, 0.869927, 0.963708, 0.0, 0.0, 0.0, 0.0, 22.578681000000003, 14.936381, 15.182515, 18.468899, 16.196503, 14.74656, 2.5978760000000003, 1.626755, 0.0, 0.0, 0.0, 0.0, 19.860733, 17.175225, 17.854823, 15.32335, 15.463053, 13.035642, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.66654, 19.448089, 17.503957999999997, 14.021732, 16.616972, 15.405708, 11.383151, 2.012989, 0.0, 0.0, 0.0, 0.0, 10.639394, 22.033980999999997, 16.482802, 17.222418, 16.121134, 14.730546, 0.0, 1.565246, 0.351481, 0.0, 0.0, 0.0, 9.533204, 22.97091, 17.418032, 16.647037, 15.199985999999999, 16.479248, 17.59538, 3.68351, 0.0, 0.0, 0.0, 0.0, 2.473911, 23.390389, 18.231482999999997, 15.665976, 18.835745, 15.563086, 2.309082, 1.043264, 0.724582, 0.0, 0.0, 0.0, 0.0, 20.363362, 18.93945, 14.974253999999998, 14.516489, 17.41579, 16.590446999999998, 12.462887, 1.6142910000000001, 0.0, 0.0, 0.0, 0.0, 18.105431, 17.066273, 17.904849, 16.427104, 16.267496, 8.024478, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.062125, 17.765486, 15.514085, 14.804003000000002, 17.488198, 15.729511, 16.140389, 16.183366, 2.376205, 21.279802, 23.581005, 21.603399, 20.795482, 23.727944, 20.052611, 23.557464, 6.109756, 0.475732, 0.0, 0.0, 0.542999]

pcie_in_bw =[]
for j in rank0_in_bw_list:
    for i in range(306):
        pcie_in_bw.append(j)

pcie_out_bw = []
for j in rank0_out_bw_list:
    for i in range(306):
        pcie_out_bw.append(j)

# exec(open(directory + 'granite-rank2-pci8/statistics/pre_dealloc.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]

motiv1 = {"Eviction from GPU" : pcie_out_bw, "Fetching to GPU" : pcie_in_bw}
ax = Figure.add_subplot(211)

# if selection == 0:
#     # motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
#     motiv1 = {"After Local Pass" : offloaded_total}
# elif selection == 1:
#     motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
# elif selection == 2:
#     motiv1 = {"Before Local Pass" : liveness_total}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "inpeal, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
# ax = Figure.add_subplot(412)
plot_timeline(ax, motiv1, "mem_consumption_bert", "GPU Kernel Index in one training iteration", " ", markevery=1, legend=True, yscale_log=False)



# exec(open(directory + 'granite-rank4-pci8/statistics/pre_dealloc.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]

# motiv1 = {"Average Fetch Bandwidth" : pcie_in_bw}
# ax = Figure.add_subplot(212)

# if selection == 0:
#     # motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
#     motiv1 = {"After Local Pass" : offloaded_total}
# elif selection == 1:
#     motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
# elif selection == 2:
#     motiv1 = {"Before Local Pass" : liveness_total}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "inp
# ax = Figure.add_subplot(413)
# plot_timeline(ax, motiv1, "mem_consumption_bert", "GPU Kernel Index\n (a) GPU 4", " ", markevery=1, legend=True, yscale_log=False)



# exec(open(directory + 'granite-rank6-pci8/statistics/pre_dealloc.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]



# if selection == 0:
#     # motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
#     motiv1 = {"After Local Pass" : offloaded_total}
# elif selection == 1:
#     motiv1 = {"Before Local Pass" : liveness_total, "After Local Pass" : offloaded_total}
# elif selection == 2:
#     motiv1 = {"Before Local Pass" : liveness_total}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "inp
# ax = Figure.add_subplot(414)
# plot_timeline(ax, motiv1, "mem_consumption_bert", "GPU Kernel Index\n (a) GPU 6", " ", markevery=1, legend=True, yscale_log=False)



# # exec(open('../../../results/VIT/512-prefetch_lru_NNMemConsumptionLog.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
# # if selection == 0:
# #     motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 1:
# #     motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 2:
# #     motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
# # ax = Figure.add_subplot(412)
# # plot_timeline(ax, motiv1, "mem_consumption_vit", "CUDA Kernel Index\n(b) ViT-512", " ", markevery=1, yscale_log=False)



# # exec(open('../../../results/ResNet152/512-prefetch_lru_NNMemConsumptionLog.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
# # if selection == 0:
# #     motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 1:
# #     motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 2:
# #     motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
# # ax = Figure.add_subplot(413)
# # plot_timeline(ax, motiv1, "mem_consumption_resnet", "CUDA Kernel Index\n(c) ResNet152-512", " ", markevery=1, yscale_log=False)



# # exec(open('../../../results/Inceptionv3/512-prefetch_lru_NNMemConsumptionLog.py').read())
# # live = active
# # live_breakdown = active_breakdown
# # live_input = [item[0] for item in live_breakdown]
# # live_weight = [item[1] + 1 for item in live_breakdown]
# # live_intermediate = [item[2] for item in live_breakdown]
# # real = total
# # global_input = [input_size for _ in real]
# # global_weight = [global_weight for _ in real]
# # global_intermediate = [s - global_input[0] - global_weight[0] for s in real]
# # if selection == 0:
# #     motiv1 = {"weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 1:
# #     motiv1 = {"all" : real, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 2:
# #     motiv1 = {"all" : real, "input" : live_input, "weight" : live_weight, "intermediate" : live_intermediate}
# # elif selection == 3:
# #     motiv1 = {"all" : real, "input" : global_input, "weight" : global_weight, "intermediate" : global_intermediate}
# # ax = Figure.add_subplot(414)
# # plot_timeline(ax, motiv1, "mem_consumption_incept", "CUDA Kernel Index\n(d) Inceptionv3-512", " ", markevery=1, yscale_log=False)



Figure.text(-0.062, 0.42, "Migration Bandwidth \n Usage (GB/s)", rotation=90, \
    horizontalalignment='center', verticalalignment='center', \
    transform=ax.transAxes, fontsize=18)

# Figure.text(-0.062, 0.42, "Migration Bandwidth \n (GB/s)", rotation=90, \
#     horizontalalignment='center', verticalalignment='center', \
#     transform=ax.transAxes)

Figure.tight_layout(pad=1.)

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()


