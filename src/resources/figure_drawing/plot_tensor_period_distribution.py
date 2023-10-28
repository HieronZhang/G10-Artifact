import os
from typing import Tuple, Union

from matplotlib.ticker import LogLocator
from fig_common import *




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
    
    ax.scatter(traces[:, 0], traces[:, 1], color="navy", marker="o", s=10, zorder=3)

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
    
    # ax.set_xlim(ax.get_xlim())
    ax.set_xlim(1, 1e10)
    ax.set_ylim(1e3, 1e10)

    # CHANGE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


batch_sizes = [4, 8, 16, 32, 64, 128, 256]
model_names = ["OPT_1.3", "OPT_2.7", "OPT_6.7", "OPT_13"]

for batch_size in batch_sizes:
    for model_name in model_names:

        try:
            exec(open(f'../../../results/{model_name}/{batch_size}-lru_TensorPeriodLog.py').read())

            Figure = plt.figure( figsize=(8, 8) )
            PDF = PdfPages( f"output/{model_name}_bs{batch_size}_tensor_periods_distribution.pdf" )
        
            
            ax = Figure.add_subplot(221)
            plot_cost_model(np.array(sd_time), sd_size, ax, ["forestgreen", "peru", "royalblue"])
            ax.text(0.45, -0.34, f"{model_name} BS{batch_size}", \
            horizontalalignment='center', verticalalignment='center', \
            transform=ax.transAxes)

            Figure.tight_layout(pad=0.8)

            PDF.savefig(Figure, bbox_inches='tight')
            PDF.close()
            plt.close(Figure)

        except:
            print(f"{model_name} batch {batch_size} not found.")
