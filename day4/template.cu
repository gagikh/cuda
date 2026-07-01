// Day 4: CUDA Memory Types and Management
// Goal: compare pageable vs. pinned vs. unified memory for the same transfer/compute.
//
// Compile:  nvcc -arch=sm_50 day4_template.cu -o day4
// Run:      ./day4  (profile with: nsys profile ./day4)

#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 1 << 22;

__global__ void add_vectors(const double *a, const double *b, double *c, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

void run_with_pageable_memory()
{
    // TODO: allocate host buffers with plain `new`/`malloc`, device buffers with cudaMalloc,
    // cudaMemcpy in, launch add_vectors, cudaMemcpy out, time with cudaEvents.
}

void run_with_pinned_memory()
{
    // TODO: allocate host buffers with cudaMallocHost instead, otherwise same as above.
    // Compare the transfer time against run_with_pageable_memory().
}

void run_with_unified_memory()
{
    // TODO: allocate a, b, c with cudaMallocManaged, touch them directly from host,
    // launch add_vectors directly on them (no explicit cudaMemcpy needed),
    // cudaDeviceSynchronize() before reading results on host.
}

int main()
{
    run_with_pageable_memory();
    run_with_pinned_memory();
    run_with_unified_memory();

    // TODO (self-learning #2): try cudaHostRegister on a pageable buffer instead of
    // allocating pinned memory up front.

    return 0;
}
