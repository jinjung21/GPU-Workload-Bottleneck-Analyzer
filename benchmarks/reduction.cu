#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__global__ void reduce_blocks_kernel(const float* input, float* partial, int n) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) {
        sum += input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }
    shared[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__global__ void reduce_final_kernel(const float* partial, float* output, int n) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    for (int idx = tid; idx < n; idx += blockDim.x) {
        sum += partial[idx];
    }
    shared[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = shared[0];
    }
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int n = 1 << 24;
    const int block_size = 256;
    const int grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    const int iterations = 80;
    const size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t partial_bytes = static_cast<size_t>(grid_size) * sizeof(float);

    float* input = nullptr;
    float* partial = nullptr;
    float* output = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_bytes));
    CHECK_CUDA(cudaMalloc(&partial, partial_bytes));
    CHECK_CUDA(cudaMalloc(&output, sizeof(float)));
    CHECK_CUDA(cudaMemset(input, 1, input_bytes));

    reduce_blocks_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(input, partial, n);
    reduce_final_kernel<<<1, block_size, block_size * sizeof(float)>>>(partial, output, grid_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        reduce_blocks_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(input, partial, n);
        reduce_final_kernel<<<1, block_size, block_size * sizeof(float)>>>(partial, output, grid_size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = static_cast<double>(n - 1);
    const double dram_read_bytes = static_cast<double>(input_bytes + partial_bytes);
    const double dram_write_bytes = static_cast<double>(partial_bytes + sizeof(float));

    if (csv) {
        std::printf("reduction,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA parallel reduction on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("reduction runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(input));
    CHECK_CUDA(cudaFree(partial));
    CHECK_CUDA(cudaFree(output));
    return 0;
}
