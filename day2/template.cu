// Day 2: Thread Hierarchy & Execution Model
// Goal: 1D vector addition, then extend to 2D "image addition".
//
// Compile:  nvcc -arch=sm_50 day2_template.cu -o day2
// Run:      ./day2

#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: 1D vector addition kernel.
// c[i] = a[i] + b[i], with a bounds check against n.
__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    // TODO
}

// TODO 2: 2D "image addition" kernel (treat images as flat buffers of size width*height).
// Use 2D thread indexing (threadIdx.x/y, blockIdx.x/y) to compute row/col,
// then flatten to a 1D offset before reading/writing.
__global__ void image_add(const unsigned char *a, const unsigned char *b,
                           unsigned char *c, int width, int height)
{
    // TODO
}

int main()
{
    // --- Part 1: vector add ---
    const int n = 1 << 20; // not a multiple of small block sizes on purpose
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // TODO: fill d_a / d_b (e.g. via a host buffer + cudaMemcpy, or a small init kernel)

    // TODO: choose block size (try 32, 64, 128, 256) and compute grid size to cover n
    // vector_add<<<?, ?>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // --- Part 2: image add ---
    const int width = 256, height = 256;
    size_t img_bytes = width * height * sizeof(unsigned char);

    unsigned char *d_imgA, *d_imgB, *d_imgC;
    cudaMalloc(&d_imgA, img_bytes);
    cudaMalloc(&d_imgB, img_bytes);
    cudaMalloc(&d_imgC, img_bytes);

    // TODO: dim3 block(...), grid(...); image_add<<<grid, block>>>(...)

    cudaFree(d_imgA);
    cudaFree(d_imgB);
    cudaFree(d_imgC);

    return 0;
}
