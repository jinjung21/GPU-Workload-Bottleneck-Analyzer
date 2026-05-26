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

__global__ void gemv_kernel(const float* matrix, const float* vector, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    const float* row_ptr = matrix + static_cast<size_t>(row) * cols;
    for (int col = 0; col < cols; ++col) {
        sum += row_ptr[col] * vector[col];
    }
    output[row] = sum;
}

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int rows = 4096;
    const int cols = 4096;
    const int iterations = 20;
    const size_t matrix_elems = static_cast<size_t>(rows) * cols;
    const size_t matrix_bytes = matrix_elems * sizeof(float);
    const size_t vector_bytes = static_cast<size_t>(cols) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows) * sizeof(float);

    float* matrix = nullptr;
    float* vector = nullptr;
    float* output = nullptr;
    CHECK_CUDA(cudaMalloc(&matrix, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&vector, vector_bytes));
    CHECK_CUDA(cudaMalloc(&output, output_bytes));
    CHECK_CUDA(cudaMemset(matrix, 1, matrix_bytes));
    CHECK_CUDA(cudaMemset(vector, 1, vector_bytes));

    dim3 block(128);
    dim3 grid((rows + block.x - 1) / block.x);
    gemv_kernel<<<grid, block>>>(matrix, vector, output, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemv_kernel<<<grid, block>>>(matrix, vector, output, rows, cols);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = static_cast<double>(2) * rows * cols;
    const double dram_read_bytes = static_cast<double>(matrix_bytes + vector_bytes);
    const double dram_write_bytes = static_cast<double>(output_bytes);

    if (csv) {
        std::printf("gemv,%.6f,%.0f,%.0f,%.0f,regular,Measured CUDA matrix-vector multiplication on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("gemv runtime_ms=%.6f rows=%d cols=%d\n", runtime_ms, rows, cols);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(matrix));
    CHECK_CUDA(cudaFree(vector));
    CHECK_CUDA(cudaFree(output));
    return 0;
}
