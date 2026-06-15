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

__global__ void saxpy_kernel(const float* x, float* y, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

int main(int argc, char** argv) {
    BenchmarkArgs args = parse_benchmark_args(argc, argv);
    const int n = choose_arg(args.n, 1 << 24);
    const int iterations = choose_arg(args.iterations, 50);
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* x = nullptr;
    float* y = nullptr;
    CHECK_CUDA(cudaMalloc(&x, bytes));
    CHECK_CUDA(cudaMalloc(&y, bytes));
    CHECK_CUDA(cudaMemset(x, 1, bytes));
    CHECK_CUDA(cudaMemset(y, 2, bytes));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    saxpy_kernel<<<grid, block>>>(x, y, 2.0f, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        saxpy_kernel<<<grid, block>>>(x, y, 2.0f, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = static_cast<double>(2 * n);
    const double dram_read_bytes = static_cast<double>(2 * bytes);
    const double dram_write_bytes = static_cast<double>(bytes);

    if (args.csv) {
        std::printf("saxpy,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA SAXPY streaming workload on GPU n=%d\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes, n);
    } else {
        std::printf("saxpy runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(y));
    return 0;
}
