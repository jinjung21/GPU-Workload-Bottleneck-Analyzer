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

__global__ void init_indices_kernel(int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = (idx * 9973) & (n - 1);
    }
}

__global__ void random_gather_kernel(const float* table, const int* indices, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = table[indices[idx]] * 1.0001f + 0.5f;
    }
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int n = 1 << 24;
    const int iterations = 30;
    const size_t float_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t index_bytes = static_cast<size_t>(n) * sizeof(int);

    float* table = nullptr;
    float* out = nullptr;
    int* indices = nullptr;
    CHECK_CUDA(cudaMalloc(&table, float_bytes));
    CHECK_CUDA(cudaMalloc(&out, float_bytes));
    CHECK_CUDA(cudaMalloc(&indices, index_bytes));
    CHECK_CUDA(cudaMemset(table, 1, float_bytes));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    init_indices_kernel<<<grid, block>>>(indices, n);
    random_gather_kernel<<<grid, block>>>(table, indices, out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        random_gather_kernel<<<grid, block>>>(table, indices, out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = static_cast<double>(2 * n);
    const double dram_read_bytes = static_cast<double>(float_bytes + index_bytes);
    const double dram_write_bytes = static_cast<double>(float_bytes);

    if (csv) {
        std::printf("random_gather,%.6f,%.0f,%.0f,%.0f,irregular,Measured CUDA random gather on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("random_gather runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(table));
    CHECK_CUDA(cudaFree(out));
    CHECK_CUDA(cudaFree(indices));
    return 0;
}
