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


## 2. Running Execution Time Profiling and Tensor Liveness Analysis for A DNN Model
This section describes the steps to how to configure and run the profiling and tensor liveness analysis of one static DNN model. We will use OPT-1.3 (batch_size=4) as an example.

### 2.1 Gnenrating Configuration Files
The first step is to generate appropriate config files. We provide an example config file `src/configs/ResNet50/64-profile_analysis.config`:

```bash
output_folder           ../results/ResNet50/pytorch64
is_pytorch              1
is_profiling            1    

nn_model_input_forward    ../frontend/Pytorch_ResNet50/fx_graph_transformed_forward.py
nn_model_input_backward    ../frontend/Pytorch_ResNet50/fx_graph_transformed_backward.py
nn_model_input_weight    ../frontend/Pytorch_ResNet50/fx_graph_transformed_weight.py

GPU_memory_size_GB        40
```

If you need to profile first, please keep `is_profiling` as 1. The `output_folder` is where you want to store the profiling results.


### 2.2 Launching the Profiling

To launch the tool, directly use the corresponding config file and use `tensor_liveness_analysis` to run it:
```bash
./tensor_liveness_analysis "$relative_path_to_config_file"
    # e.g.,  ./tensor_liveness_analysis configs/ResNet50/64-profile_analysis.config
```

You will see all the profiling results in the `output_folder`.


### 2.3 Program Outputs

After running one single experiment, the program will generate several files under the corresponding `results/` directory. These files will have the following directory hierarchy:
```bash
results/"$model_name"
├── pytorch$batchSize
│   └── statistics
│       ├── kernels.config
│       ├── tensors.config
│       ├── liveness_analysis.config
│       └── ...
├── pytorch$batchSize-$additionalParameter_TensorPeriodLog.py
└── pytorch$batchSize-$additionalParameter_NNMemConsumptionLog.py
```

## kernels.config
This file contains information for each scheduled GPU kernel in one training iteration. Users should see something like this:
```bash
____________________________________________________________________
Kernel ID: 6, Name: torch.ops.aten.mul.Tensor(sub, rsqrt)
Input Tensors:
tensor327, Name: sub, Is global?: 0: Size in byte: 205520896, Dimensions: f32[64, 64, 112, 112]
tensor326, Name: rsqrt, Is global?: 0: Size in byte: 256, Dimensions: f32[1, 64, 1, 1]
Output Tensors:
tensor328, Name: mul, Is global?: 0: Size in byte: 205520896, Dimensions: f32[64, 64, 112, 112]
____________________________________________________________________
Kernel ID: 7, Name: torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
Input Tensors:
tensor324, Name: getitem_1, Is global?: 0: Size in byte: 256, Dimensions: f32[1, 64, 1, 1]
Output Tensors:
tensor329, Name: squeeze, Is global?: 0: Size in byte: 256, Dimensions: f32[64]
____________________________________________________________________
Kernel ID: 8, Name: torch.ops.aten.mul.Tensor(squeeze, 0.1)
Input Tensors:
tensor329, Name: squeeze, Is global?: 0: Size in byte: 256, Dimensions: f32[64]
Output Tensors:
tensor330, Name: mul_1, Is global?: 0: Size in byte: 256, Dimensions: f32[64]
____________________________________________________________________
```

## tensors.config
This file contains information for all tensors. Users should see something like this:
```bash
tensor315, Name: primals_316, Is global?: 1: Size in byte: 2048, 
tensor316, Name: primals_317, Is global?: 1: Size in byte: 8, 
tensor317, Name: primals_318, Is global?: 1: Size in byte: 8192, 
tensor318, Name: primals_319, Is global?: 1: Size in byte: 8192, 
tensor319, Name: primals_320, Is global?: 1: Size in byte: 8, 
tensor320, Name: primals_321, Is global?: 1: Size in byte: 38535168, 
tensor321, Name: convolution, Is global?: 0: Size in byte: 205520896, 
tensor322, Name: add, Is global?: 0: Size in byte: 8, 
tensor323, Name: getitem, Is global?: 0: Size in byte: 256, 
tensor324, Name: getitem_1, Is global?: 0: Size in byte: 256, 
tensor325, Name: add_1, Is global?: 0: Size in byte: 256, 
tensor326, Name: rsqrt, Is global?: 0: Size in byte: 256, 
tensor327, Name: sub, Is global?: 0: Size in byte: 205520896, 
tensor328, Name: mul, Is global?: 0: Size in byte: 205520896, 
tensor329, Name: squeeze, Is global?: 0: Size in byte: 256, 
tensor330, Name: mul_1, Is global?: 0: Size in byte: 256, 
tensor331, Name: mul_2, Is global?: 0: Size in byte: 256, 
tensor332, Name: add_2, Is global?: 0: Size in byte: 256, 
tensor333, Name: mul_3, Is global?: 0: Size in byte: 256, 
tensor334, Name: mul_4, Is global?: 0: Size in byte: 256, 
```

## liveness_analysis.config 
This file includes the results of the Tensor Vitality Analysis. The inactive periods ("Hiding intervals" in this file) and live periods are shown for every tensor. Users should see something like this:
```bash
_____________________________________________________________________________
tensor1161, Name: add_242, Is global?: 0: Size in byte: 25690112, 
Liveness: Birth: 793, Death: 811.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tensor1161, Name: add_242, Is global?: 0: Size in byte: 25690112, 
Inactive Periods:
Period 0: 794--------810
Estimated Time:1061.83us
_____________________________________________________________________________
_____________________________________________________________________________
tensor1162, Name: convolution_46, Is global?: 0: Size in byte: 25690112, 
Liveness: Birth: 794, Death: 1040.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tensor1162, Name: convolution_46, Is global?: 0: Size in byte: 25690112, 
Inactive Periods:
Period 0: 795--------796
Estimated Time:16.8342us
Period 1: 797--------799
Estimated Time:32.2283us
Period 2: 800--------1039
Estimated Time:11699.6us
_____________________________________________________________________________
_____________________________________________________________________________
```

## TensorPeriodLog.py
This file is for drawing the figure for the inactive periods of tensors (See fig 3-4 in G10).

## NNMemConsumptionLog.py
This file is for drawing the figure for the memory consumption of DNN training (See fig 2 in G10).



