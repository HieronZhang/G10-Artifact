import matplotlib.pyplot as plt
import numpy as np
from fig_common import *

title1 = "output/roofline"

PDF1 = PdfPages(title1 + ".pdf")

# Input data
# modelname = ["llama-70B-BS8-L4096", "llama-8B-BS16-L4096"]
# modelname = ["llama-70B-BS8-L4096", "llama-8B-BS16-L4096", "granite-3B-BS32-L1024", "granite-8B-BS16-L1024", "mistral-7B-BS16-L1024", "gpt2-40B-BS16-L1024", "BertL-BS128-L512"]
modelname = [ "llama-8B-BS8-L2048", "llama-8B-BS8-L3072", "llama-8B-BS8-L4096","llama-8B-BS16-L1024", "llama-8B-BS16-L2048", "llama-8B-BS16-L3072", "llama-8B-BS16-L4096"]
# ranks = ["rank0", "rank1", "rank2", "rank3"]
ranks = ["rank0"]

# Time and memory stride for each model

name = []
time_stride_arr = []  # in microseconds (us)
memory_stride_arr = []  # in bytes (B)

for model_i in [0, 1, 2, 3, 4 , 5, 6]:
    for rank_i in range(1):
        rank = ranks[rank_i]
        # rank = "rank0"
        filename = "../../../results/" + modelname[model_i] + "/" + f"rank{rank_i}_" + "NNMemConsumptionLog.py"
        exec(open(filename).read())
        name.append(modelname[model_i] + "-" + f"stage{rank_i}")
        time_stride_arr.append(time_stride)
        memory_stride_arr.append(memory_stride)

# Convert units for plotting
time_in_seconds = np.array(time_stride_arr) / 1e6  # Convert time to seconds
memory_in_gb = np.array(memory_stride_arr) / 1e9  # Convert memory to GB

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 5))

# Scatter plot
for i in range(len(name)):
    ax.scatter(time_in_seconds[i], memory_in_gb[i], label=name[i], s=100)

# Plot lines for bandwidths
x_vals = np.linspace(min(time_in_seconds) * 0.5, max(time_in_seconds) * 1.5, 100)  # Define x-axis range
bandwidth_16gbps = 16 * x_vals  # 16 GB/s
bandwidth_32gbps = 32 * x_vals  # 32 GB/s

ax.plot(x_vals, bandwidth_16gbps, label="16 GB/s (PCIe Gen3)", color='#598570', linestyle="--", linewidth=2.0)
ax.plot(x_vals, bandwidth_32gbps, label="32 GB/s (PCIe Gen4)", color='#23a8eb', linestyle="--", linewidth=2.0)

# Labeling axes
ax.set_xlabel('Exe. Time Needed to Reach Peak Mem. Consumption (s)', fontsize=15)
ax.set_ylabel('"Peak Memory Consumption" (GB)', fontsize=17)
# ax.set_title("Scatter Plot with Bandwidth Lines", fontsize=14)

# Set log scale if needed (optional)
ax.set_xscale("log")
ax.set_yscale("log")

# Add legend
ax.legend(fontsize=9.5)

# Add grid
ax.grid(which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.tight_layout()
PDF1.savefig(fig)
PDF1.close()
