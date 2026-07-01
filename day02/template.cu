// Day 2: Thread Hierarchy & Execution Model
// Goal: 1D vector addition, a grid-stride loop variant, then 2D "image addition".
//
// Compile:  nvcc -arch=sm_50 day02_template.cu -o day02
// Run:      ./day02

#include <cstdio>
#include <cuda_runtime.h>
#include "../common/cuda_check.h"

// TODO 1: 1D vector addition kernel.
// c[i] = a[i] + b[i], with a bounds check against n.
__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    // TODO
}

// TODO 2 (grid-stride loop): same computation as vector_add, but launched
// with a FIXED grid/block size chosen once (see main()), not sized to
// exactly cover `n`. Each thread processes multiple elements, striding by
// the total number of threads in the grid (gridDim.x * blockDim.x) each
// time round the loop.
//
// Why this matters: vector_add above needs its grid size recomputed for
// every different `n`, and silently does the wrong thing if you get that
// arithmetic wrong. A grid-stride loop handles ANY `n` -- including one
// you don't know until runtime -- with the same fixed launch configuration.
// This is the idiomatic pattern for real (non-toy) CUDA code.
__global__ void vector_add_grid_stride(const float *a, const float *b, float *c, int n)
{
    // TODO: int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //       int stride = gridDim.x * blockDim.x;
    //       for (int i = idx; i < n; i += stride) c[i] = a[i] + b[i];
}

// TODO 3: 2D "image addition" kernel (treat images as flat buffers of size width*height).
// Use 2D thread indexing (threadIdx.x/y, blockIdx.x/y) to compute row/col,
// then flatten to a 1D offset before reading/writing.
__global__ void image_add(const unsigned char *a, const unsigned char *b,
                           unsigned char *c, int width, int height)
{
    // TODO
}

int main()
{
    // --- Part 1: vector add, grid sized exactly to n ---
    const int n = 1 << 20; // not a multiple of small block sizes on purpose
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // TODO: fill d_a / d_b (e.g. via a host buffer + cudaMemcpy, or a small init kernel)

    // TODO: choose block size (try 32, 64, 128, 256) and compute grid size to cover n
    // vector_add<<<?, ?>>>(d_a, d_b, d_c, n);
    // CUDA_CHECK_LAST_ERROR();

    // --- Part 1b: same computation, fixed launch size, grid-stride loop ---
    // Note this launch configuration doesn't depend on `n` at all -- try
    // changing `n` above to something much larger and confirm this still works
    // without touching the numbers below.
    const int threads = 256;
    const int blocks = 256; // fixed, deliberately not derived from n
    vector_add_grid_stride<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // --- Part 2: image add ---
    const int width = 256, height = 256;
    size_t img_bytes = width * height * sizeof(unsigned char);

    unsigned char *d_imgA, *d_imgB, *d_imgC;
    CUDA_CHECK(cudaMalloc(&d_imgA, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_imgB, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_imgC, img_bytes));

    // TODO: dim3 block(...), grid(...); image_add<<<grid, block>>>(...); CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaFree(d_imgA));
    CUDA_CHECK(cudaFree(d_imgB));
    CUDA_CHECK(cudaFree(d_imgC));

    return 0;
}
