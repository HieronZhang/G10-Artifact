# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statsFiguresUtil import *
from figureUtils import *

inc_base = True

# hatch_color = "#78756e"
hatch_color = "white"

if inc_base:
  color_arr = colors_new1[::-1]
  hatch_arr = ["", "-", "/", "\\", "x", ""]
else:
  color_arr = colors_roller_4
  hatch_arr = ["", "-", "x", "", "/", "\\"]


def get_ticks_covering_ymax(y_max, n_ticks=5):
    raw_step = y_max / (n_ticks - 1)
    magnitude = 10 ** int(np.floor(np.log10(raw_step)))
    normalized = raw_step / magnitude

    # Choose a "nice" step
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10

    step = nice * magnitude
    max_tick = step * (int(np.ceil(y_max / step)))
    ticks = np.arange(0, max_tick + step, step)
    return ticks
  

# plot font size options
plt.rc("font", size=11)
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=15)
plt.rc("legend", fontsize=18)
plt.rc("hatch", color="white")
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["hatch.linewidth"] = 1.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.family': 'serif'})

def draw_overall_perf_subplot(ax, data_file, include_base=True):
    if include_base:
        bar_width = 0.10
        horiz_margin = 0.56
        horiz_major_tick = 0.7
        color_arr = colors_new1[::-1]
        hatch_arr = ["", "-", "/", "\\", "x", ""]
    else:
        bar_width = 0.135
        horiz_margin = 0.6
        horiz_major_tick = 0.7
        color_arr = colors_roller_4
        hatch_arr = ["", "-", "x", "", "/", "\\"]

    try:
        with open(data_file, "r") as f:
            lines = f.read()
    except Exception as e:
        print(f"Error reading {data_file}: {e}")
        return

    sections = lines.strip().split("\n\n")
    settings = [s.strip() for s in sections[0].split("|")]
    workloads = []
    data_array = np.zeros((len(sections) - 1, len(settings)))
    x_tick_array = np.zeros((len(sections) - 1, len(settings)))

    y_max = 0
    for section_idx, section in enumerate(sections[1:]):
        lines = section.strip().split("\n")
        workload = lines[0].strip()
        workloads.append(workload)
        data_array[section_idx, :] = np.array([float(d) for d in lines[1].split()])
        y_max = max(y_max, np.max(data_array[section_idx, :]))
        x_tick_array[section_idx, :] = section_idx * horiz_major_tick + (np.arange(len(settings)) - (len(settings) - 1) / 2) * bar_width

    for j, setting in enumerate(settings):
      for i in range(len(workloads)):
          val = data_array[i, j]
          x = x_tick_array[i, j]

          if val == 0:
              ax.plot(x, 10, 'x', color='red', markersize=10, markeredgewidth=2, zorder=4)
          else:
              ax.bar(x, val, color=color_arr[j], width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[j], zorder=3)
              ax.bar(x, val, color="none", width=bar_width, edgecolor="white", linewidth=0.8, zorder=3)
    
    
    ax.set_xticks(np.arange(len(workloads)) * horiz_major_tick)
    ax.set_xticklabels([w.replace("|", "\n") for w in workloads])
    ax.set_xlim([-horiz_margin * horiz_major_tick, (len(workloads) - 1 + horiz_margin) * horiz_major_tick])

    yticks = get_ticks_covering_ymax(y_max, n_ticks=5)
    ax.set_yticks(yticks)
    ax.set_ylim([0, 1.1 * y_max])

    ax.yaxis.grid(zorder=0)
    ax.hlines(0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], zorder=9, color='black', linewidth=1)

    return settings  # return for legend



fig, axes = plt.subplots(1, 3, figsize=(27, 4))
plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.12)

data_files = [
    "overall_pref_seq/llama8b.txt",
    "overall_pref_seq/granite8b.txt",
    "overall_pref_seq/llama70b.txt",
]

titles = [
    "Llama3-8B",
    "Granite-code-base-8B",
    "Llama3-70B"]

for i in range(3):
    settings = draw_overall_perf_subplot(axes[i], data_files[i], include_base=inc_base)
    axes[i].set_title(titles[i], fontsize=24)

from matplotlib.patches import Patch
legend_handles = []

for j, setting in enumerate(settings):
    patch = Patch(facecolor=color_arr[j], edgecolor=hatch_color, hatch=hatch_arr[j], label=setting)
    legend_handles.append(patch)

fig.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12),
           ncol=len(settings), columnspacing=2.5, fontsize=24)


# Common y-axis label
fig.text(0.075, 0.5, "Training Throughput\n        (token/s)", va='center', rotation='vertical', fontsize=24)

# Shared legend (only once)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=14)

fig.savefig("output/OverallPerfAllModels_seq.png", bbox_inches='tight')
fig.savefig("output/OverallPerfAllModels_seq.pdf", bbox_inches='tight')







# fig, ax0 = plt.subplots(1, 1, figsize=(14, 2.5))
# plt.subplots_adjust(top = 1.02, bottom=0.01, hspace=0.6, wspace=0.20)

# if inc_base:
#   data_file = f"overall_performance/all.base.txt"
# else:
#   data_file = f"overall_performance/all.txt"

# if inc_base:
#   bar_width = 0.105
#   horiz_margin = 0.6
#   horiz_major_tick = 0.75
# else:
#   bar_width = 0.135
#   horiz_margin = 0.6
#   horiz_major_tick = 0.7
# try:
#   with open(data_file, "r") as f:
#     lines = f.read()
# except Exception as e:
#   exit(1)

# settings = []
# workloads = []
# sections = lines.strip().split("\n\n")
# settings = [setting for setting in sections[0].split("|")]
# data_array = np.zeros((len(sections) - 1, len(settings)))
# # settings = settings[:-1]
# x_tick_array = np.zeros((len(sections) - 1, len(settings)))
# for section_idx, section in enumerate(sections[1:]):
#   lines = section.strip().split("\n")
#   workload = lines[0].strip()
#   workloads.append(workload)
#   data_array[section_idx, :] = np.array([float(data) for data in lines[1].split()])
#   y_max = 0
#   y_max = np.max(data_array[section_idx, :]) if np.max(data_array[section_idx, :]) > y_max else y_max
  
#   x_tick_array[section_idx, :] = section_idx * horiz_major_tick + (np.arange(len(settings)) - (len(settings) - 1) / 2) * bar_width
# for j, setting in enumerate(settings):
#   if inc_base:
#     base = -1
#   else:
#     base = 0
#   ax0.bar(x_tick_array[:, j], data_array[:, j], color=color_arr[j], width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[j], label=setting, zorder=3)
# # for j, setting in enumerate(settings):
#   ax0.bar(x_tick_array[:, j], data_array[:, j], color="none", width=bar_width, edgecolor="white", linewidth=0.8, zorder=3)
# # for x_tick, data in zip(x_tick_array[:, 0], data_array[:, 0]):
# #   ax.text(x_tick, 1.05, f"{data:5.2f} kops/s", ha="center", va="bottom", rotation=90, fontsize=10)

# ax0.set_xticks(np.arange(len(workloads)) * horiz_major_tick)
# ax0.set_xticklabels([workload.replace("|", "\n") for workload in workloads])
# ax0.set_xlim([-horiz_margin * horiz_major_tick, (len(workloads) - 1 + horiz_margin) * horiz_major_tick])
# if inc_base:
#   y_limit = 1.4 * y_max

#   yticks = get_ticks_covering_ymax(y_max, n_ticks=5)

#   ax0.set_yticks(yticks)
#   ax0.set_ylim([0, 1.4 * y_max])
# # else:
# #   ax0.set_yticks(np.arange(0, 1.8, 0.25))
# #   ax0.set_ylim([0, 2.1])
# ax0.set_ylabel("Training Throughput\n(token/s)", fontsize=18)
# # ax0.yaxis.set_label_coords(-0.06, 0.4)
# # ax0.hlines(y=1, xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1], colors="grey", linestyles="--")
# ax0.yaxis.grid(zorder=0)
# ax0.hlines(0, xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1], zorder=9, color='black', linewidth=1)

# handles, labels = ax0.get_legend_handles_labels()
# if inc_base:
#   legend = ax0.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, columnspacing=2.5)
# else:
#   legend = ax0.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, columnspacing=1.5)

# # shift = max([t.get_window_extent().width for t in legend.get_texts()])
# # for t in legend.get_texts():
# #     t.set_position(((shift - t.get_window_extent().width) / 2 - 6, 0))

# # plt.show()

# # extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(0.85, 1.45)
# # x_len, y_len = extent.x1 - extent.x0, extent.y1 - extent.y0
# # if inc_base:
# #   extent.y0 -= y_len * 0.08
# # else:
# #   extent.y0 -= y_len * 0.02
# # extent.y1 -= y_len * 0.11
# # extent.x0 -= x_len * 0.04
# # extent.x1 -= x_len * 0.02
# # figname = list(net_name_translation.values())[model]
# # fig.savefig(f"OverallPerf{figname}.png", bbox_inches=extent)
# if inc_base:
#   fig.savefig(f"output/OverallPerfNew.png", bbox_inches='tight')
#   fig.savefig(f"output/OverallPerfNew.pdf", bbox_inches='tight')
# else:
#   fig.savefig(f"output/OverallPerfRealImpl.png", bbox_inches='tight')
#   fig.savefig(f"output/OverallPerfRealImpl.pdf", bbox_inches='tight')

# %%
