// Day 3: Warp-Level Execution and Control Flow
// Goal: large vector addition with timing, then BGR->grayscale conversion.
//
// Compile:  nvcc -arch=sm_50 day3_template.cu -o day3
// Run:      ./day3

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

constexpr int N = 1 << 22;

// TODO 1: vector addition kernel (as in the README code walkthrough).
__global__ void add_vectors(const double *a, const double *b, double *c, int n)
{
    // TODO
}

// TODO 2: BGR -> grayscale kernel.
// Input: interleaved BGR buffer (3 bytes/pixel). Output: 1 byte/pixel grayscale.
// gray = 0.114*B + 0.587*G + 0.299*R
__global__ void bgr_to_gray(const unsigned char *bgr, unsigned char *gray,
                             int width, int height)
{
    // TODO
}

int main()
{
    // --- Part 1: large vector add + timing ---
    size_t bytes = N * sizeof(double);
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // TODO: initialize d_A, d_B (host buffer + cudaMemcpy, or init kernel)

    const int thr_per_blk = 256;
    const int blk_in_grid = (N + thr_per_blk - 1) / thr_per_blk;

    // Host-side timing with <chrono> (device-side cudaEvent timing comes in Day 6/7).
    auto t0 = std::chrono::high_resolution_clock::now();
    add_vectors<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("add_vectors: %.3f ms\n", ms);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // --- Part 2: BGR -> grayscale ---
    const int width = 256, height = 256;
    unsigned char *d_bgr, *d_gray;
    cudaMalloc(&d_bgr, width * height * 3);
    cudaMalloc(&d_gray, width * height);

    // TODO: dim3 block(...), grid(...); bgr_to_gray<<<grid, block>>>(...)

    cudaFree(d_bgr);
    cudaFree(d_gray);

    return 0;
}
