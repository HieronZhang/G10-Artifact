import os
from typing import Tuple, Union

from matplotlib.ticker import LogLocator
from fig_common import *

from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import matplotlib.cm as cm


Figure = plt.figure( figsize=(10.5, 8) )
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
    
    


def plot_cost_model_with_frequency(
    times, sizes, ax: plt.Axes, color_map=cm.Purples, 
    ylabel: bool = True, log_x: bool = True, log_y: bool = True, 
    y_lim: Tuple[float, float] = None, bins: Union[int, Tuple[int, int]] = (100, 100)
):
    """
    Scatter plot with frequency-based color intensity (heatmap-like).
    Ensures all points fall within the histogram range.
    """
    # Calculate custom bin edges to fully cover the data range
    x_min, x_max = np.min(times), np.max(times)
    y_min, y_max = np.min(sizes), np.max(sizes)
    
    # Create bin edges with a small buffer to avoid edge cases
    xedges = np.linspace(x_min, x_max, bins[0] + 1)
    yedges = np.linspace(y_min, y_max, bins[1] + 1)

    # Create a 2D histogram
    freq, xedges, yedges = np.histogram2d(times, sizes, bins=(xedges, yedges))

    # Transpose the frequency matrix to align with x-y indexing
    freq = freq.T  

    # Normalize frequencies using a logarithmic scale
    norm = LogNorm(vmin=1, vmax=freq.max())  # Log scale normalization
    color_mapper = cm.ScalarMappable(norm=norm, cmap=color_map)

    # Find bin indices for each point
    x_idx = np.clip(np.digitize(times, xedges) - 1, 0, len(xedges) - 2)
    y_idx = np.clip(np.digitize(sizes, yedges) - 1, 0, len(yedges) - 2)


    # Assign colors to each point
    colors = [
        color_mapper.to_rgba(freq[x_idx[i], y_idx[i]])
        for i in range(len(times))
    ]

    # Scatter plot with assigned colors
    ax.scatter(times, sizes, color=colors, marker="o", s=10, zorder=3)
    
    # Define the x values for the lines (same range as x-axis)
    x_vals = np.array([1, 4e8])  # From 1 µs to 400M µs

    # Compute y values using size = bandwidth * time
    y_vals_3GB = 3e3 * x_vals  # 3GB/s
    y_vals_12GB = 12e3 * x_vals  # 12GB/s

    # Plot the lines
    # ax.plot(x_vals, y_vals_3GB, color='red', linestyle='--', linewidth=2, label="3 GB/s")
    ax.plot(x_vals, y_vals_12GB, color='orange', linestyle='--', linewidth=2, label="12 GB/s")

    # Add legend
    ax.legend(loc="upper left", fontsize=12, frameon=True)


    # Configure axes
    ax.set_xlabel("Inactive Time ($\mu$s)")
    if ylabel:
        ax.set_ylabel("Size (byte)")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    if y_lim:
        ax.set_ylim(y_lim[0] or ax.get_ylim()[0], y_lim[1] or ax.get_ylim()[1])
    ax.set_xlim(ax.get_xlim())

    # Add a colorbar to indicate frequency
    cbar = plt.colorbar(color_mapper, ax=ax)
    cbar.set_label("Frequency (log scale)")



    
exec(open('../../../results/llama-70B-BS8-L4096/rank0_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank0_TensorPeriodLog.py').read())
ax = Figure.add_subplot(221)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cost_model_with_frequency(
    np.array(sd_time), sd_size, ax, 
    color_map=cm.Purples, bins=(20,20), y_lim=(None, 4e11)
)
ax.text(0.45, -0.34, "(a) llama-70B-GPU0(Stage-0)", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

exec(open('../../../results/granite-8B-BS16-L1024/rank0_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank1_TensorPeriodLog.py').read())
ax = Figure.add_subplot(222)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cost_model_with_frequency(
    np.array(sd_time), sd_size, ax, 
    color_map=cm.Purples, bins=(20,20), y_lim=(None, 4e11)
)
ax.text(0.45, -0.34, "(b) Granite-8B-GPU0(Stage-0)", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

exec(open('../../../results/BertL-BS128-L512/rank0_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank2_TensorPeriodLog.py').read())
ax = Figure.add_subplot(223)
# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cost_model_with_frequency(
    np.array(sd_time), sd_size, ax, 
    color_map=cm.Purples, bins=(20,20), y_lim=(None, 4e11)
)
ax.text(0.45, -0.34, "(c) Bert-Large-GPU0(Stage-0)", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

exec(open('../../../results/gpt4-40B-BS16-L1024/rank0_TensorPeriodLog.py').read())
# exec(open('../../../results/granite-8B-BS16-L1024/rank3_TensorPeriodLog.py').read())
ax = Figure.add_subplot(224)

# plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"], y_lim=(None, 4e8))
plot_cost_model_with_frequency(
    np.array(sd_time), sd_size, ax, 
    color_map=cm.Purples, bins=(20,20), y_lim=(None, 4e11)
)
ax.text(0.45, -0.34, "(d) GPT2-40B-GPU0(Stage-0)", \
  horizontalalignment='center', verticalalignment='center', \
  transform=ax.transAxes)
ax.set_xlim(1, 4e8)

Figure.tight_layout(pad=0.8)

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()
