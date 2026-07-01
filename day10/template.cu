// Day 10: Practical Algorithms
// Goal: naive matrix multiplication, then Hamming distance matching.
//
// Compile:  nvcc -arch=sm_50 day10_template.cu -o day10
// Run:      ./day10

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 16

// TODO 1: naive matrix multiplication. C = A * B, all N x N.
__global__ void matmul_naive(const float *A, const float *B, float *C, int n)
{
    // TODO: int row = ..., col = ...;
    //       float sum = 0; for (k in 0..n) sum += A[row*n+k] * B[k*n+col];
    //       C[row*n+col] = sum;
}

// TODO 2 (self-learning #2): shared-memory tiled matrix multiplication.
// Compare timing against matmul_naive for the same n.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // TODO: loop over tiles, load As/Bs, __syncthreads(), accumulate, __syncthreads()
}

// TODO 3 (self-learning #3): Hamming distance between two 32-bit descriptors using __popc.
__device__ int hamming_distance(unsigned int a, unsigned int b)
{
    // TODO: return __popc(a ^ b);
    return 0;
}

// TODO 4 (self-learning #4): for each query descriptor, find the index of the
// closest descriptor in a reference set (smallest Hamming distance).
__global__ void match_descriptors(const unsigned int *queries, int num_queries,
                                   const unsigned int *refs, int num_refs,
                                   int *best_match_idx)
{
    // TODO
}

int main()
{
    const int n = 256;
    size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // TODO: fill d_A, d_B with test data

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
