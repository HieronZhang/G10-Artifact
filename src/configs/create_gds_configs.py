#TODO: change these directories when using the script
import os
import sys

settings = ["8b_16_2k_4ubs", "8b_32_2k_4ubs", "8b_64_1k_8ubs", "8b_64_2k_4ubs", "8b_64_4k_2ubs", "8b_128_2k_4ubs", "70b_8_2k_1ubs", "70b_16_1k_1ubs", "70b_16_2k_1ubs", "70b_16_3k_1ubs", "70b_16_4k_1ubs", "70b_32_2k_1ubs", "70b_64_2k_1ubs"
            , "granite_8b_bs16_seq4k_ubs4", "granite_8b_bs32_seq4k_ubs4", "granite_8b_bs64_seq4k_ubs4", "granite_8b_bs64_seq2k_ubs8", "granite_8b_bs64_seq8k_ubs2", "granite_8b_bs128_seq4k_ubs4"]

speedups = ["1", "5", "10", "10", "10", "40", "40", "80", "80", "80", "80", "160", "320", "20", "40", "40", "40", "40", "40"]

ranks = ["0", "1"]

gpu_sizes = ["84", "82", "80", "78", "76", "74"]

cpu_sizes = ["0", "1024"]

for i in range(len(settings)):
    # create the directories first
    model_i = settings[i]
    speedup = speedups[i]
    for rank in ranks:
        for cpu_size in cpu_sizes:
            for gpu_size in gpu_sizes:
                output_dir = "torchtitan_" + rank + "_" + model_i + "_gpu" + gpu_size + "_cpu" + cpu_size
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

SSD_PCIe_bandwidth_GBps 16
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20


CPU_PCIe_bandwidth_GBps 52
CPU_memory_line_GB      {cpu_size}


PCIe_latency_us         5

delta_parameter         0.5
"""
                    print(content, file=fout)
