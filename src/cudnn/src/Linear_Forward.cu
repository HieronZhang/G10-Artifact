#include <assert.h>
#include <vector>
#include "Linear_Forward.h"
#include "cublas_v2.h"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)


extern std::unordered_map<long, float*> allocation_map; 


// __global__ void sgemm(int m, int n, int k, 
//         const float *input, const float *weight, const float *bias, float *output) {

//     // Macros for accessing flattened matrices
//     #define input(row, col)  input[(row) + (col) * m]
//     #define weight(row, col) weight[(row) * n + (col)]
//     #define output(row, col) output[(row) + (col) * m]

//     __shared__ float B_shared[TILE_SZ_RATIO][TILE_SZ_B];
//     int row = blockIdx.x * TILE_SZ_A + threadIdx.x;

//     int n_iter_num = ceil(n * 1.0 / TILE_SZ_B);
//     int k_iter_num = ceil(k * 1.0 / TILE_SZ_RATIO);
//     for (int n_iter = 0; n_iter < n_iter_num; n_iter++) {
//         for (int k_iter = 0; k_iter < k_iter_num; k_iter++) {
//         // load weight tile into shared memory, weight is transposed
//         int shared_start_row = k_iter * TILE_SZ_RATIO;
//         int shared_start_col = n_iter * TILE_SZ_B;
//         int shared_row_offset = threadIdx.x / TILE_SZ_B;
//         int shared_col_offset = threadIdx.x % TILE_SZ_B;

//         if (shared_start_row + shared_row_offset < k && shared_start_col + shared_col_offset < n) {
//             B_shared[shared_row_offset][shared_col_offset] = 
//                     weight(shared_start_row + shared_row_offset, shared_start_col + shared_col_offset);
//         } else {
//             B_shared[shared_row_offset][shared_col_offset] = 0;
//         }

//         __syncthreads();

//         for (int j = 0; j < TILE_SZ_B; j++) {
//             float output_cumulative = (n_iter == 0 && k_iter == 0) ? 0 : bias[j];
//             for (int i = 0; i < TILE_SZ_RATIO; i++) {
//                 if (row < m && shared_start_col + j < n && shared_start_row + i < k) {
//                     output_cumulative += input(row, k_iter * TILE_SZ_RATIO + i) * B_shared[i][j];
//                 }
//             }
//             output(row, shared_start_col + j) += output_cumulative;
//         }
        
//         __syncthreads();
//         }
//     }
// }

Linear_Forward::Linear_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    // 4. h_in          5. h_out
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    h_in       = args[4];   h_out      = args[5];
    input_indicator = args[6]; weight_indicator = args[7]; output_indicator = args[8];
    input_ratio = args[9]; output_ratio = args[10];
    reshape = (long) input_n * input_c * input_h * input_w / h_in;

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) reshape * h_in * sizeof(float)));
        CUDA_CALL(cudaMalloc(&weight_data, (long) h_in * h_out * sizeof(float)));
        CUDA_CALL(cudaMalloc(&bias_data, (long) h_out * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) reshape * h_out * sizeof(float)));
        GPUFillRand(input_data, (long) reshape * h_in * sizeof(float));
        GPUFillRand(weight_data, (long) h_in * h_out * sizeof(float));
        GPUFillRand(bias_data, (long) h_out * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Linear_Forward::~Linear_Forward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(weight_data));
        CUDA_CALL(cudaFree(bias_data));
        CUDA_CALL(cudaFree(output_data));
    }
}

float Linear_Forward::Run() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    if (is_UVM) {

        
        if (allocation_map.find(input_indicator) != allocation_map.end()) {
            input_data = allocation_map[input_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&input_data, (long) reshape * h_in * sizeof(float)));
            allocation_map[input_indicator] = input_data;
        }

        if (allocation_map.find(weight_indicator) != allocation_map.end()) {
            weight_data = allocation_map[weight_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&weight_data, (long) h_in * h_out * sizeof(float)));
            allocation_map[weight_indicator] = weight_data;
        }
        
        if (allocation_map.find(output_indicator) != allocation_map.end()) {
            output_data = allocation_map[output_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&output_data, (long) reshape * h_out * sizeof(float)));
            allocation_map[output_indicator] = output_data;
        }

        // CUDA_CALL(cudaMallocManaged(&input_data, (long) reshape * h_in * sizeof(float)));
        // CUDA_CALL(cudaMallocManaged(&weight_data, (long) h_in * h_out * sizeof(float)));
        // CUDA_CALL(cudaMallocManaged(&bias_data, (long) h_out * sizeof(float)));
        // CUDA_CALL(cudaMallocManaged(&output_data, (long) reshape * h_out * sizeof(float)));
        // CPUFillRand(input_data, (long) reshape * h_in * sizeof(float));
        // CPUFillRand(weight_data, (long) h_in * h_out * sizeof(float));
        // CPUFillRand(bias_data, (long) h_out * sizeof(float));

        // GPUFillRand(input_data, (long) reshape * h_in * sizeof(float) * input_ratio);
        // GPUFillRand(weight_data, (long) h_in * h_out * sizeof(float) * input_ratio);
        // GPUFillRand(bias_data, (long) h_out * sizeof(float) * input_ratio);
        // GPUFillRand(output_data, (long) reshape * h_out * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    // dim3 BlockSize(TILE_SZ_A, 1, 1);
    // dim3 GridSize(ceil((double) reshape / TILE_SZ_A), 1, 1);

    float alpha = 1;
    float beta = 0;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    // sgemm<<<GridSize, BlockSize>>>(reshape, h_out, h_in, input_data, weight_data, bias_data, output_data);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, reshape, h_out, h_in, &alpha, input_data, reshape, weight_data, h_in, &beta, output_data, reshape);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // if (is_UVM) {
    //     CUDA_CALL(cudaFree(input_data));
    //     CUDA_CALL(cudaFree(weight_data));
    //     CUDA_CALL(cudaFree(bias_data));
    //     CUDA_CALL(cudaFree(output_data));
    // }
    return milliseconds;
}





BatchMatMul_Forward::BatchMatMul_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    // 4. inputB_dim3(h_in)         5. inputB_dim4(h_out)
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    h_in       = args[4];   h_out      = args[5];
    input1_indicator = args[6]; input2_indicator = args[7]; output_indicator = args[8];
    input_ratio = args[9]; output_ratio = args[10];
    

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data_A, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float)));

        CUDA_CALL(cudaMalloc(&output_data, (long) input_n * input_c * input_h * h_out * sizeof(float)));
        GPUFillRand(input_data_A, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float));

    }
    cudaDeviceSynchronize();
}

BatchMatMul_Forward::~BatchMatMul_Forward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data_A));
        CUDA_CALL(cudaFree(input_data_B));

        CUDA_CALL(cudaFree(output_data));
    }
}

float BatchMatMul_Forward::Run() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    if (is_UVM) {

        if (allocation_map.find(input1_indicator) != allocation_map.end()) {
            input_data_A = allocation_map[input1_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&input_data_A, (long) input_n * input_c * input_h * input_w * sizeof(float)));
            allocation_map[input1_indicator] = input_data_A;
        }

        if (allocation_map.find(input2_indicator) != allocation_map.end()) {
            input_data_B = allocation_map[input2_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float)));
            allocation_map[input2_indicator] = input_data_B;
        }

        if (allocation_map.find(output_indicator) != allocation_map.end()) {
            output_data = allocation_map[output_indicator];
        }
        else
        {
            CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * input_c * input_h * h_out * sizeof(float)));
            allocation_map[output_indicator] = output_data;
        }

        // CUDA_CALL(cudaMallocManaged(&input_data_A, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        // CUDA_CALL(cudaMallocManaged(&input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float)));

        // CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * input_c * input_h * h_out * sizeof(float)));
        // CPUFillRand(input_data_A, (long)  input_n * input_c * input_h * input_w * sizeof(float));
        // CPUFillRand(input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float));


        // GPUFillRand(input_data_A, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        // GPUFillRand(input_data_B, (long) input_n * input_c * h_in * h_out * sizeof(float) * input_ratio);

        // GPUFillRand(output_data, (long) input_n * input_c * input_h * h_out * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    // dim3 BlockSize(TILE_SZ_A, 1, 1);
    // dim3 GridSize(ceil((double) reshape / TILE_SZ_A), 1, 1);

    float alpha = 1;
    float beta = 0;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    // sgemm<<<GridSize, BlockSize>>>(reshape, h_out, h_in, input_data_A, input_data_B, bias_data, output_data);
    // 
    long A2, A3, A4;
    A2 = input_h;
    A3 = input_w;
    A4 = h_out;

    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, A4, A2, A3,
        &alpha, input_data_B, A4, A3*A4, input_data_A, A3, A2*A3,
        &beta, output_data, A4, A2*A4, input_n*input_c);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        // CUDA_CALL(cudaFree(input_data_A));
        // CUDA_CALL(cudaFree(input_data_B));

        // CUDA_CALL(cudaFree(output_data));
    }
    return milliseconds;
}
