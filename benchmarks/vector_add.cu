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

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int n = 1 << 24;
    const int iterations = 50;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    CHECK_CUDA(cudaMalloc(&a, bytes));
    CHECK_CUDA(cudaMalloc(&b, bytes));
    CHECK_CUDA(cudaMalloc(&c, bytes));
    CHECK_CUDA(cudaMemset(a, 1, bytes));
    CHECK_CUDA(cudaMemset(b, 2, bytes));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vector_add_kernel<<<grid, block>>>(a, b, c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        vector_add_kernel<<<grid, block>>>(a, b, c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = static_cast<double>(n);
    const double dram_read_bytes = static_cast<double>(2 * bytes);
    const double dram_write_bytes = static_cast<double>(bytes);

    if (csv) {
        std::printf("vector_add,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA vector addition on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("vector_add runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 0;
}
