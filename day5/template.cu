// Day 5: Memory Conflicts and Shared Memory
// Goal: shared-memory tiled 2D filter (box blur as a starting point, then Sobel).
//
// Compile:  nvcc -arch=sm_50 day5_template.cu -o day5
// Run:      ./day5

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define RADIUS 1 // 3x3 filter

// TODO: tiled 2D filter kernel.
// 1. Load a (TILE_DIM + 2*RADIUS)^2 tile (including halo) into shared memory.
// 2. __syncthreads() before reading neighbors.
// 3. Each thread computes one output pixel from its shared-memory neighborhood.
__global__ void tiled_filter(const unsigned char *in, unsigned char *out,
                              int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];

    // TODO: compute global x, y and load into `tile`, including halo cells
    // TODO: __syncthreads();
    // TODO: compute filtered value from `tile` neighborhood, write to `out`
}

// TODO (self-learning #3): Sobel filter kernel, reusing the tiling approach above.
__global__ void sobel_filter(const unsigned char *in, unsigned char *out,
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

    // TODO: fill d_in with test image data

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    tiled_filter<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
