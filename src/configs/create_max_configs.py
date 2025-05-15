#TODO: change these directories when using the script
import os
import sys

settings = ["70b_8_2k_2ubs", "70b_16_1k_2ubs", "70b_16_2k_2ubs", "70b_16_3k_2ubs", "70b_16_4k_2ubs", "70b_32_2k_2ubs", "70b_32_2k_4ubs", "70b_16_1k_4ubs"]

speedups = [ "5", "10", "10", "10", "10", "20", "20", "20"]

ranks = ["0", "1"]

gpu_sizes = ["82", "80", "78", "76"]

cpu_sizes = [["561","561","561","561","561","561", "561", "561"],
             ["463","463","463","463","463","463", "463", "463"]]

for i in range(len(settings)):
    # create the directories first
    model_i = settings[i]
    speedup = speedups[i]
    cpu_usage = ""
    for rank in ranks:

            if rank == "0":
                cpu_size = cpu_sizes[0][i]
            else:
                cpu_size = cpu_sizes[1][i]
            cpu_usage = "max"
            for gpu_size in gpu_sizes:
                output_dir = "torchtitan_" + rank + "_" + model_i + "_gpu" + gpu_size + "_" + cpu_usage
                directory_name = "data_" + model_i
                filename = model_i + "/" + f"rank{rank}" + f"_gpu{gpu_size}" + f"_cpu{cpu_size}"
                directory = os.path.dirname(filename)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                with open(filename+".config", 'w') as fout:
                    content = f"""
output_folder           ../results/{output_dir}
input_directory         ../{directory_name}/{rank}/semantics.in
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

SSD_PCIe_bandwidth_GBps {"16" if cpu_size == "0" else "11"}
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20


CPU_PCIe_bandwidth_GBps 54
CPU_memory_line_GB      {cpu_size}


PCIe_latency_us         5

delta_parameter         0.5
"""
                    print(content, file=fout)
