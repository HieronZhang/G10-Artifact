#TODO: change these directories when using the script
directory_name = ["/home/haoyang/Code_repos/py_mem_snap/llama_8B_bs16_seq1024/3B_tp_pp_1f1b/memory_snapshot/iteration_10","/home/haoyang/Code_repos/py_mem_snap/granite_3B_bs64_seq2048", "/home/haoyang/Code_repos/py_mem_snap/opt_run/3B_tp_pp_1f1b/memory_snapshot/iteration_10" ]

model_name = ["llama", "granite", "opt"]

ranks = ["rank0", "rank2", "rank4", "rank6"]

pcie_array = [4, 6, 8,12,16,20,24,28,32]

for model_i in [0, 1, 2]:
    # for rank in ranks:
    for pcie in pcie_array:
        filename = model_name[model_i] + "-" + f"pci{pcie}"
        with open(filename+".config", 'w') as fout:
            content = f"""
output_folder           ../results/{filename}
input_directory              {directory_name[model_i]}
is_simulation           1


num_iteration           2
num_threads             128

stat_output_file        sim_result

is_UVM                  1
use_prefetch            1
eviction_policy         LRU

system_latency_us       45

GPU_memory_size_GB      40
GPU_frequency_GHz       1.2
GPU_PCIe_bandwidth_GBps {pcie}
GPU_malloc_uspB         0.000000814
GPU_free_uspB           0

SSD_PCIe_bandwidth_GBps {pcie}
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20


CPU_PCIe_bandwidth_GBps {pcie}
CPU_memory_line_GB      0
PCIe_batch_size_page    50

NVLink_bandwidth_GBps   600

PCIe_latency_us         5

delta_parameter         0.5
"""
            print(content, file=fout)
