#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
CXXBIN="${CXXBIN:-/usr/bin/g++-9}"

mkdir -p build

echo "Using nvcc: $NVCC"
echo "Using host compiler: $CXXBIN"

"$NVCC" -O3 -std=c++14 -ccbin "$CXXBIN" benchmarks/vector_add.cu -o build/vector_add
"$NVCC" -O3 -std=c++14 -ccbin "$CXXBIN" benchmarks/random_gather.cu -o build/random_gather
"$NVCC" -O3 -std=c++14 -ccbin "$CXXBIN" benchmarks/matrix_mul_tiled.cu -o build/matrix_mul_tiled

echo "Built CUDA benchmarks in build/"
