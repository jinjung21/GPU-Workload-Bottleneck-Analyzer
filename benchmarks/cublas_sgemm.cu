#include <cublas_v2.h>
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

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,    \
                         __LINE__, static_cast<int>(status));                    \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    bool csv = argc > 1 && std::strcmp(argv[1], "--csv") == 0;
    const int n = 2048;
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
    CHECK_CUDA(cudaMemset(c, 0, matrix_bytes));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        a,
        n,
        b,
        n,
        &beta,
        c,
        n));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha,
            a,
            n,
            b,
            n,
            &beta,
            c,
            n));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float runtime_ms = elapsed_ms / iterations;

    const double flops = 2.0 * static_cast<double>(n) * n * n;
    const double dram_read_bytes = 2.0 * static_cast<double>(matrix_bytes);
    const double dram_write_bytes = static_cast<double>(matrix_bytes);

    if (csv) {
        std::printf("cublas_sgemm,%.6f,%.0f,%.0f,%.0f,regular,Measured cuBLAS SGEMM on GPU\n",
                    runtime_ms, flops, dram_read_bytes, dram_write_bytes);
    } else {
        std::printf("cublas_sgemm runtime_ms=%.6f n=%d\n", runtime_ms, n);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 0;
}
