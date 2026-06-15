#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "benchmark_args.h"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int BLOCK_SIZE = 1024;

__global__ void block_scan_kernel(const float* input, float* output, int n) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = idx < n ? input[idx] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float value = 0.0f;
        if (tid >= offset) {
            value = shared[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            shared[tid] += value;
        }
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

int main(int argc, char** argv) {
    BenchmarkArgs args = parse_benchmark_args(argc, argv);
    const int n = choose_arg(args.n, 1 << 24);
    const int iterations = choose_arg(args.iterations, 40);
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* input = nullptr;
    float* output = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes));
    CHECK_CUDA(cudaMalloc(&output, bytes));
    CHECK_CUDA(cudaMemset(input, 1, bytes));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);
    block_scan_kernel<<<grid, block>>>(input, output, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        block_scan_kernel<<<grid, block>>>(input, output, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    double additions_per_block = 0.0;
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        additions_per_block += static_cast<double>(BLOCK_SIZE - offset);
    }
    const double blocks = static_cast<double>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const double flops = additions_per_block * blocks;
    const double dram_read_bytes = static_cast<double>(bytes);
    const double dram_write_bytes = static_cast<double>(bytes);

    if (args.csv) {
        std::printf("scan,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA block-local prefix scan on GPU n=%d\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes, n);
    } else {
        std::printf("scan runtime_ms=%.6f n=%d block_size=%d\n", runtime_ms, n, BLOCK_SIZE);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(input));
    CHECK_CUDA(cudaFree(output));
    return 0;
}
