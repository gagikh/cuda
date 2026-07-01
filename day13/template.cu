// Day 13: Cache Behavior and Optimization
// Goal: apply __ldg and bank-conflict-free shared memory to an existing kernel.
//
// Compile:  nvcc -arch=sm_50 day13_template.cu -o day13
// Run:      ./day13

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define RADIUS 1

// Baseline: the Day 5 tiled filter, unmodified.
__global__ void tiled_filter_baseline(const unsigned char *in, unsigned char *out,
                                       int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
    // TODO: same body as Day 5's tiled_filter
}

// TODO 1 (self-learning #1): same filter, but read `in` through __ldg() since it's
// read-only for the duration of the kernel.
__global__ void tiled_filter_ldg(const unsigned char *__restrict__ in, unsigned char *out,
                                  int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
    // TODO: load into `tile` using __ldg(&in[idx]) instead of in[idx]
}

// TODO 2 (self-learning #2): apply a swizzled shared-memory access pattern to
// remove bank conflicts (see the swizzling reference in the README).
__global__ void tiled_filter_swizzled(const unsigned char *in, unsigned char *out,
                                       int width, int height)
{
    // TODO
}

int main()
{
    const int width = 512, height = 512;
    size_t bytes = width * height * sizeof(unsigned char);

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // TODO: fill d_in with test data

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_filter_baseline<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_baseline = 0.0f;
    cudaEventElapsedTime(&ms_baseline, start, stop);
    printf("baseline: %.3f ms\n", ms_baseline);

    cudaEventRecord(start);
    tiled_filter_ldg<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_ldg = 0.0f;
    cudaEventElapsedTime(&ms_ldg, start, stop);
    printf("__ldg:    %.3f ms\n", ms_ldg);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
