# A DNN Profiling and Tensor Liveness Analysis Tool (from G10)

## 1. Installation

### 1.1 Downloading the Repository
Use the following command to download the tool:

```bash
git clone git@github.com:HieronZhang/???.git
```

### 1.2 Installation

Install the following dependencies:
```bash
sudo apt install flex bison tmux python3-pip
pip3 install matplotlib networkx pandas PyPDF2
```

Build the profiling and tensor liveness profiling tool (the output executable is named `tensor_liveness_analysis`):
```bash
cd src/
make clean
make -j
```

Note that to do profiling, users need to first correctly install `CUDA` (11.0 and higher) Tool-kits with `cudnn` and `cublas` libraries. Before doing profiling, please make sure that the CUDA code generation part of our framework is built:
```bash
cd src/cudnn
make clean && make
```


## 2. Running Execution Time Profiling of A Static DNN Model
This section describes the steps to how to configure and run the profiling of one static DNN model. We will use OPT-1.3 (batch_size=4) as an example.

### 2.1 Gnenrating Configuration Files
The first step is to generate appropriate config files (for profiling). We provide an example config file `src/configs/OPT_1.3/4-profile.config`:

```bash
output_folder           ../results/OPT_1.3/cudnn4         
is_profiling            1

trans_borden            355
batch_size              4
seq_len                 128

nn_model_input_file     ../frontend/Nets/OPT_1.3.txt
```

For profiling, please keep `is_profiling` as 1. The `output_folder` is where you want to store the profiling results.


### 2.2 Launching the Profiling

To launch the profiling, directly find its corresponding config file and use `tensor_liveness_analysis` to run it:
```bash
./tensor_liveness_analysis "$relative_path_to_config_file"
    # e.g.,  ./tensor_liveness_analysis configs/OPT_1.3/4-profile.config
```

You will see all the profiling results in the `output_folder`.



## 3. Running Tensor Liveness Analysis of A Static DNN Model
After doing the profiling, with the execution time of every DNN program kernel, we can then do the Tensor Liveness Analysis.

### 3.1 Gnenrating Configuration Files
The first step is to generate appropriate config files (for analysis). We provide an example config file `src/configs/OPT_1.3/4-analysis.config`:

```bash
output_folder           ../results/OPT_1.3/4-analysis
is_simulation             1

batch_size                4
seq_len                   128

nn_model_input_file       ../frontend/Nets/OPT_1.3.txt
orig_kernel_time_file     ../results/OPT_1.3/cudnn4.txt
workspace_size_file       ../results/OPT_1.3/cudnn4Workspace.txt
stat_output_file          sim_result

GPU_memory_size_GB        40
GPU_frequency_GHz         1.2
```

For tensor liveness analysis, please keep `is_simulation` as 1. The `output_folder` is where you want to store the analysis results.

Note: the `orig_kernel_time_file` and `workspace_size_file` should match with the profiled execution time file and the workspace file generated when doing profiling.

### 2.2 Launching the Liveness Analysis

To launch the analysis, directly find its corresponding config file and use `tensor_liveness_analysis` to run it:
```bash
./tensor_liveness_analysis "$relative_path_to_config_file"
    # e.g.,  ./tensor_liveness_analysis configs/OPT_1.3/4-analysis.config
```

The program will execute the Tensor Vitality Analysis Algorithms. The results will be generated in `results/` directory. 


### 2.3 Program Outputs

After running one single experiment, the program will generate several files under the corresponding `results/` directory. These files will have the following directory hierarchy:
```bash
results/"$model_name"
├── $batchSize-analysis
│   └── statistics
│       ├── layers.config
│       ├── kernels.config
│       ├── tensors.config
│       ├── interval.config
│       ├── prefetch_guide.config
│       └── ...
├── $batchSize-analysis-$additionalParameter_TensorPeriodLog.py
└── $batchSize-analysis-$additionalParameter_NNMemConsumptionLog.py
```


## layers.config
This file contains analyzed model operator/layer information. Users should see something like this:
```bash
______________________________________________________________________________
Layer ID:3; Name:Conv2d (512,32,149,149)
Next Layers:
Next Layer 0
Layer ID:4; Name:BatchNorm2d (512,32,147,147)
Previous Layers:
Previous Layer 0
Layer ID:2; Name:ReLU (512,32,149,149)
Input Tensor: tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Output Tensor: tensor20 Is weight (global)?: 0, Size in byte: 1416167424, Range:9828384768--11244552192
d_Input Tensor: tensor23 Is weight (global)?: 0, Size in byte: 1454964736, Range:11244589056--12699553792
d_Output Tensor: tensor25 Is weight (global)?: 0, Size in byte: 1416167424, Range:14115721216--15531888640
Weight Tensor: tensor21 Is weight (global)?: 1, Size in byte: 36864, Range:16384--53248
d_Weight Tensor: tensor22 Is weight (global)?: 0, Size in byte: 36864, Range:11244552192--11244589056
______________________________________________________________________________
```
## kernels.config
This file contains information for each scheduled GPU kernel in one training iteration. Users should see something like this:
```bash
____________________________________________________________________
Kernel ID: 4, Name: Conv2d_Forward
Parent Layer ID:3; Name:Conv2d
Execution Time: 16564224
(512,32,149,149)
Input Tensors:
tensor21 Is weight (global)?: 1, Size in byte: 36864, Range:16384--53248
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
Output Tensors:
tensor1891 Is weight (global)?: 0, Size in byte: 2871173120, Range:166677065728--169548238848
tensor20 Is weight (global)?: 0, Size in byte: 1416167424, Range:9828384768--11244552192
____________________________________________________________________
```

## tensors.config
This file contains information for all tensors. Users should see something like this:
```bash
tensor0 Is weight (global)?: 0, Size in byte: 549281792, Range:0--549281792
tensor1 Is weight (global)?: 0, Size in byte: 1454964736, Range:549281792--2004246528
tensor2 Is weight (global)?: 1, Size in byte: 4096, Range:0--4096
tensor3 Is weight (global)?: 0, Size in byte: 4096, Range:2004246528--2004250624
tensor4 Is weight (global)?: 0, Size in byte: 549281792, Range:2004250624--2553532416
tensor5 Is weight (global)?: 0, Size in byte: 1454964736, Range:2553532416--4008497152
tensor6 Is weight (global)?: 0, Size in byte: 1454964736, Range:4008497152--5463461888
```

## interval.config 
This file includes the results of the Tensor Vitality Analysis. The inactive periods ("Hiding intervals" in this file) and live periods are shown for every tensor. Users should see something like this:
```bash
_______________________________________________________________
tensor8 Is weight (global)?: 0, Size in byte: 4194304, Range:437018624--441212928
Liveness: Birth: 2, Death: 3085.
____________________________________________________________
tensor8 Is weight (global)?: 0, Size in byte: 4194304, Range:437018624--441212928
Inactive Periods:
Period 0: 4--------5
Estimated Time:7495.68
Period 1: 6--------26
Estimated Time:709045
Period 2: 27--------3084
Estimated Time:9.65945e+07
_______________________________________________________________
```

## TensorPeriodLog.py
This file is for drawing the figure for the inactive periods of tensors (See fig 3-4 in G10).

## NNMemConsumptionLog.py
This file is for drawing the figure for the memory consumption of DNN training (See fig 2 in G10).



