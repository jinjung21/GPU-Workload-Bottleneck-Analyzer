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

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_kernel(const float* input, float* output, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        tile[threadIdx.y + row][threadIdx.x] = input[(y + row) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        output[(y + row) * width + x] = tile[threadIdx.x][threadIdx.y + row];
    }
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int width = 4096;
    const int n = width * width;
    const int iterations = 40;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* input = nullptr;
    float* output = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes));
    CHECK_CUDA(cudaMalloc(&output, bytes));
    CHECK_CUDA(cudaMemset(input, 1, bytes));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(width / TILE_DIM, width / TILE_DIM);
    transpose_kernel<<<grid, block>>>(input, output, width);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        transpose_kernel<<<grid, block>>>(input, output, width);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = 0.0;
    const double dram_read_bytes = static_cast<double>(bytes);
    const double dram_write_bytes = static_cast<double>(bytes);

    if (csv) {
        std::printf("matrix_transpose,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA tiled matrix transpose on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("matrix_transpose runtime_ms=%.6f width=%d\n", runtime_ms, width);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(input));
    CHECK_CUDA(cudaFree(output));
    return 0;
}
