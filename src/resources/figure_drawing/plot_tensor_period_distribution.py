import os
from typing import Tuple, Union

from matplotlib.ticker import LogLocator
from fig_common import *

from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from figureUtils import *

Figure = plt.figure( figsize=(18, 4.5) )
PDF = PdfPages( "output/tensor_periods_distribution.pdf" )



def plot_cost_model(times, sizes, ax: plt.Axes, color_list: List[str], ylabel: bool = True, log_x: bool = True, log_y: bool = True, y_lim: Tuple[float, float] = None, plot_line_slope: float = 1717.986918):
    '''
    color_list[0] for T10, color_list[1:] for baseline_points;
    baseline_points: [poplib (mem, time), roller (mem, time)]
    '''

    
    data = np.array([times, sizes]).T

    traces = np.array(data)
    # ax.scatter(list(range(traces.shape[0])), traces[:, 0], color="lightgreen", marker="o", s=10, label="Measured")
    # ax.plot(list(range(traces.shape[0])), traces[:, 1], color="navy", label="Predicted")
    
    # plot y=x
    #ax.plot([0, 1e8], np.array([0, 1e8]) * plot_line_slope, color="grey", linestyle="--", linewidth=1, zorder=2)
    
    ax.scatter(traces[:, 0], traces[:, 1], color="#098eb3", marker="o", s=10, zorder=3)

    ax.set_xlabel("Inactive Time ($\mu$s)")
    if ylabel:
        ax.set_ylabel("Size (byte)")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # set xtick labels the same as ytick labels
    ax.xaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
    ax.yaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    if y_lim:
        ax.set_ylim(y_lim[0] or ax.get_ylim()[0], y_lim[1] or ax.get_ylim()[1])
    
    ax.set_xlim(ax.get_xlim())
    
def plot_cdf_by_size_ranges(
    times, sizes, ax: plt.Axes, 
    ylabel: bool = True, log_x: bool = True,
    size_bins=None, labels=None, colors=None, linestyles=None
):
    if size_bins is None:
        size_bins = [
            (0, 1 << 20),
            (1 << 20, 10 << 20),
            (10 << 20, 100 << 20),
            (100 << 20, 1 << 30),
            (1 << 30, float("inf"))
        ]
    if labels is None:
        labels = ["<1MB", "1–10MB", "10–100MB", "100MB–1GB", ">1GB"]
    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # custom dash style for the last one

    total_count = len(times)

    for (lower, upper), label, color, ls in zip(size_bins, labels, colors, linestyles):
        mask = (sizes >= lower) & (sizes < upper)
        filtered_times = times[mask]
        count = len(filtered_times)

        if count == 0:
            continue

        sorted_times = np.sort(filtered_times)
        cdf = np.linspace(0, 100, count, endpoint=True)

        pct = 100.0 * count / total_count
        label_with_pct = f"{label} ({pct:.1f}%)"

        ax.plot(sorted_times, cdf, label=label_with_pct, color=color, linestyle=ls, linewidth=3)

    # Configure axes
    ax.set_xlabel("Inactive Time ($\mu$s)")
    if ylabel:
        ax.set_ylabel("CDF (%)")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="grey")

    # Always place legend in the upper-left corner
    ax.legend(loc="upper left", fontsize=10, frameon=True)



    
exec(open('../../../results/llama-8B-BS128-L2048/_pcie4_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank0_TensorPeriodLog.py').read())
ax = Figure.add_subplot(141)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
size_bins = [
            (0, 1 << 20),
            (1 << 20, 10 << 20),
            (10 << 20, 100 << 20),
            (100 << 20, 200 << 20),
            (200 << 20, float("inf"))
        ]
plot_cdf_by_size_ranges(np.array(sd_time), np.array(sd_size), ax, True, True, size_bins, ["<1MB", "1–10MB", "10–100MB", "100MB–200MB", ">200MB"])

ax.text(0.45, -0.34, "(a) Llama3-8B", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

exec(open('../../../results/gpt4-40B-BS16-L1024/_pcie64_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank1_TensorPeriodLog.py').read())
ax = Figure.add_subplot(142)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cdf_by_size_ranges(np.array(sd_time), np.array(sd_size), ax)

ax.text(0.45, -0.34, "(b) GPT2-40B", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

exec(open('../../../results/llama-70B-BS64-L2048/_pcie16_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank2_TensorPeriodLog.py').read())
ax = Figure.add_subplot(143)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cdf_by_size_ranges(np.array(sd_time), np.array(sd_size), ax)
ax.text(0.45, -0.34, "(c) Llama3-70B", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e9)

exec(open('../../../results/T5-11B-BS256-L512/_pcie4_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank3_TensorPeriodLog.py').read())
ax = Figure.add_subplot(144)

size_bins = [
            (0, 1 << 20),
            (1 << 20, 10 << 20),
            (10 << 20, 50 << 20),
            (50 << 20, 100 << 20),
            (200 << 20, float("inf"))
        ]
plot_cdf_by_size_ranges(np.array(sd_time), np.array(sd_size), ax, True, True, size_bins, ["<1MB", "1–10MB", "10–50MB", "50MB–100MB", ">100MB"])


# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
# plot_cdf_by_size_ranges(np.array(sd_time), np.array(sd_size), ax)
ax.text(0.45, -0.34, "(d) T5-11B", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

Figure.tight_layout(pad=0.8)

PDF.savefig(Figure, bbox_inches='tight')
Figure.savefig(f"output/tensor_periods_distribution.png", bbox_inches='tight')
PDF.close()
