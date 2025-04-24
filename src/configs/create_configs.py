#TODO: change these directories when using the script
# directory_name = ["../../semantics/70b-8batch-4096len/", "../../semantics/8b-16batch-1024len/", "../../semantics/8b-16batch-2048len/", "../../semantics/8b-16batch-3072len/", "../../semantics/8b-16batch-4096len/", "../../semantics/granite3b-32batch-1024len/", "../../semantics/granite8b-16batch-1024len/", "../../semantics/mistral7b-16batch-1024len/", "../../semantics/gpt4-40b_bs16_seq1024/", "../../semantics/llama8b_run/bs24_seq2048/", "../../semantics/llama8b_run/bs24_seq1024/", "../../semantics/llama8b_run/bs32_seq1024/", 
#                   "../../semantics/llama8b_run/bs8_seq1024/", "../../semantics/llama8b_run/bs8_seq2048/", "../../semantics/llama8b_run/bs8_seq3072/", "../../semantics/llama8b_run/bs8_seq4096/", "../../semantics/bertl-128batch-512seqlen/"]

directory_name = ["../../semantics/70b-8batch-4096len/", "../../semantics/8b-16batch-2048len/",  "../../semantics/granite3b-32batch-1024len/", "../../semantics/granite8b-16batch-1024len/", "../../semantics/mistral7b-16batch-1024len/", "../../semantics/gpt4-40b_bs16_seq1024/", 
                    "../../semantics/bertl-128batch-512seqlen/", "../../semantics/t5_11b_32batch_512len/"]


# model_name = ["llama-70B-BS8-L4096", "llama-8B-BS16-L1024", "llama-8B-BS16-L2048", "llama-8B-BS16-L3072", "llama-8B-BS16-L4096", "granite-3B-BS32-L1024", "granite-8B-BS16-L1024", "mistral-7B-BS16-L1024", "gpt4-40B-BS16-L1024", "llama-8B-BS24-L2048", "llama-8B-BS24-L1024", "llama-8B-BS32-L1024", "llama-8B-BS8-L1024", "llama-8B-BS8-L2048", "llama-8B-BS8-L3072", "llama-8B-BS8-L4096", "BertL-BS128-L512"]

model_name = ["llama-70B-BS8-L4096", "llama-8B-BS16-L2048", "granite-3B-BS32-L1024", "granite-8B-BS16-L1024", "mistral-7B-BS16-L1024", "gpt4-40B-BS16-L1024", "BertL-BS128-L512", "T5-11B-BS32-L512"]


ranks = ["rank0"]

rank_names = ["liveness-rank0-step1.liveness"]

cpu_sizes = ["0", "80", "160"]

pcie_array = [4, 8, 12, 16, 20, 24]

for model_i in [0, 1, 2, 3, 4, 5, 6, 7]:
    # create the directories first
    for rank_i in range(1):
        for pcie in pcie_array:
        # for cpu_size in cpu_sizes:
            rank = ranks[rank_i]
            filename = model_name[model_i] + "/" + f"rank{rank_i}" + f"_pcie{pcie}"
            with open(filename+".config", 'w') as fout:
                content = f"""
output_folder           ../results/{filename}
input_directory         {directory_name[model_i]+rank_names[rank_i]}
is_simulation           1


num_iteration           3
num_threads             128

stat_output_file        sim_result

use_prefetch            1
eviction_policy         LRU
migration_policy        G10GDSSSD

system_latency_us       45

GPU_memory_size_GB      20
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


PCIe_latency_us         5

delta_parameter         0.5
"""
                print(content, file=fout)
