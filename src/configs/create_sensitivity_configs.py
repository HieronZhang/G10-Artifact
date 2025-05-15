#TODO: change these directories when using the script
import os
import sys

settings = ["8b_64_4k_2ubs", "70b_16_2k_2ubs"]

speedups = ["4", "10"]

ranks = [0, 1]
rank_names = ["0", "1"]

gpu_sizes = ["82", "80", "78", "76"]

cpu_sizes = ["16", "32", "64", "128", "256", "512", "1024"]
ssd_numbers = ["1", "2", "3", "4"]

cpu_sizes_8b = {"16":[12, 4], "32": [24, 8], "64":[49, 15], "128": [98, 30], "256": [196, 60], "512": [392, 120], "1024": [783, 241]}
cpu_sizes_70b = {"128": [70, 58], "256": [140, 116], "512": [280, 232], "1024": [561, 463], "64": [35, 29], "32": [17, 14], "16": [8, 7]}

ssd_bws = {"1": "3", "2": "6", "3": "9", "4": "11"}


for i in range(len(settings)):
    # create the directories first
    model_i = settings[i]
    speedup = speedups[i]
    cpu_usage = ""
    cpu_size_per_rank = ""
    for cpu_size in cpu_sizes:
        for ssd_number in ssd_numbers:
            for gpu_size in gpu_sizes:
                for rank in ranks:
                    if model_i == "8b_64_4k_2ubs":
                        cpu_size_per_rank = str(cpu_sizes_8b[cpu_size][rank])
                        cpu_usage = cpu_size            
                    else:
                        cpu_size_per_rank = str(cpu_sizes_70b[cpu_size][rank])
                        cpu_usage = cpu_size
                    ssd_bw = ssd_bws[ssd_number]
                    output_dir = "torchtitan_" + rank_names[rank] + "_" + model_i + "_gpu" + gpu_size + "_cpu" + cpu_usage + "_ssd" + ssd_number
                    directory_name = "data_" + model_i
                    filename = model_i + "/" + f"rank{rank_names[rank]}" + f"_gpu{gpu_size}" + f"_cpu{cpu_usage}" + f"_ssd{ssd_number}"
                    directory = os.path.dirname(filename)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    with open(filename+".config", 'w') as fout:
                        content = f"""
output_folder           ../results/{output_dir}
input_directory         ../{directory_name}/{rank_names[rank]}/semantics.in
is_simulation           1


stat_output_file        sim_result

use_prefetch            1
algo_speedup            {speedup}

migration_policy        G10GDSSSD


GPU_memory_size_GB      {gpu_size}
GPU_frequency_GHz       1.2
GPU_PCIe_bandwidth_GBps 64
GPU_malloc_uspB         0.000000814
GPU_free_uspB           0

SSD_PCIe_bandwidth_GBps {ssd_bw}
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20


CPU_PCIe_bandwidth_GBps 54
CPU_memory_line_GB      {cpu_size_per_rank}


PCIe_latency_us         5

delta_parameter         0.5
"""
                        print(content, file=fout)
