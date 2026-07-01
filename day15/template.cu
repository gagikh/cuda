// Day 15: Stream-Ordered Memory Allocation
// Goal: replace cudaMalloc/cudaFree with stream-ordered cudaMallocAsync/cudaFreeAsync,
// then benchmark allocation overhead against the classic API.
//
// Compile:  nvcc -arch=sm_60 day15_template.cu -o day15
// Run:      ./day15

#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_vectors(const float *a, const float *b, float *c, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

void run_with_malloc_async(cudaStream_t stream, int n)
{
    float *d_a, *d_b, *d_c;

    // TODO 1: allocate d_a/d_b/d_c with cudaMallocAsync(&ptr, bytes, stream) instead of cudaMalloc.
    // TODO: launch add_vectors on `stream`.
    // TODO: free with cudaFreeAsync(ptr, stream) instead of cudaFree.
}

void benchmark_alloc_overhead(int iterations, size_t bytes)
{
    // TODO 2 (self-learning #2): time `iterations` iterations of cudaMalloc+cudaFree
    // vs. cudaMallocAsync+cudaFreeAsync for a small `bytes` allocation, using
    // cudaEvents around each loop. Which one wins, and by how much?
}

int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int n = 1 << 16;
    run_with_malloc_async(stream, n);

    benchmark_alloc_overhead(1000, 4096);

    // TODO (self-learning #3): create an explicit cudaMemPool_t with cudaDeviceGetDefaultMemPool
    // or cudaMemPoolCreate, and drive allocations on it from two different streams.

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}
