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

constexpr int TILE = 16;

__global__ void matrix_mul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += TILE) {
        int a_col = tile + threadIdx.x;
        int b_row = tile + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int n = 1024;
    const int iterations = 10;
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(float);

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    CHECK_CUDA(cudaMalloc(&a, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&b, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&c, matrix_bytes));
    CHECK_CUDA(cudaMemset(a, 1, matrix_bytes));
    CHECK_CUDA(cudaMemset(b, 2, matrix_bytes));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    matrix_mul_tiled_kernel<<<grid, block>>>(a, b, c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        matrix_mul_tiled_kernel<<<grid, block>>>(a, b, c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = 2.0 * static_cast<double>(n) * n * n;
    const double dram_read_bytes = 2.0 * static_cast<double>(n) * n * n * sizeof(float) / TILE;
    const double dram_write_bytes = static_cast<double>(matrix_bytes);

    if (csv) {
        std::printf("matrix_mul_tiled,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA tiled GEMM on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("matrix_mul_tiled runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 0;
}
