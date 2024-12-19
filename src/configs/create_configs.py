#TODO: change these directories when using the script
directory_name = ["../../semantics/70b-8batch-4096len/liveness-rank0-step1.liveness"]

model_name = ["llama-70B"]

ranks = ["rank0", "rank2", "rank4", "rank6"]

cpu_sizes = ["0", "80", "160"]

pcie_array = [4, 8, 12]

for model_i in [0]:
    # for rank in ranks:
    for pcie in pcie_array:
        for cpu_size in cpu_sizes:
            filename = model_name[model_i] + "/" + f"ssd{pcie}-cpu{cpu_size}"
            with open(filename+".config", 'w') as fout:
                content = f"""
output_folder           ../results/{filename}
input_directory         {directory_name[model_i]}
is_simulation           1


num_iteration           3
num_threads             128

stat_output_file        sim_result

use_prefetch            1
eviction_policy         LRU
migration_policy        G10GDSSSD

system_latency_us       45

GPU_memory_size_GB      80
GPU_frequency_GHz       1.2
GPU_PCIe_bandwidth_GBps 15.754
GPU_malloc_uspB         0.000000814
GPU_free_uspB           0

SSD_PCIe_bandwidth_GBps {pcie}
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20


CPU_PCIe_bandwidth_GBps 15.754
CPU_memory_line_GB      {cpu_size}
PCIe_batch_size_page    50


PCIe_latency_us         5

delta_parameter         0.5
"""
                print(content, file=fout)
