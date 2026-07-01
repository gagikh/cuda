// Day 1: CUDA Basics and Programming Model
// Goal: compile, launch, and run a minimal kernel; inspect thread/block identity.
//
// Compile:  nvcc -arch=sm_50 day1_template.cu -o day1
// Run:      ./day1

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

// TODO 1: write a kernel that prints its own block/thread index using device-side printf.
__global__ void hello_kernel()
{
    // TODO: printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// TODO 2: write a kernel that computes each thread's global index and stores it in `out`.
__global__ void global_index_kernel(int *out, int n)
{
    // TODO: compute global index = blockDim.x * blockIdx.x + threadIdx.x
    // TODO: bounds-check against n before writing
}

int main()
{
    // --- Part 1: say hello ---
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    // --- Part 2: compute + verify global indices ---
    const int n = 16;
    int *d_out = nullptr;
    int h_out[n] = {0};

    cudaMalloc(&d_out, n * sizeof(int));

    // TODO: choose a grid/block configuration that covers `n` threads
    // global_index_kernel<<<?, ?>>>(d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        printf("out[%d] = %d\n", i, h_out[i]);
    }

    cudaFree(d_out);

    // TODO (self-learning #4): time a 1-thread launch vs a many-thread launch
    // from the host side using <chrono>, e.g.:
    //
    //   auto t0 = std::chrono::high_resolution_clock::now();
    //   hello_kernel<<<1, 1>>>();
    //   cudaDeviceSynchronize();
    //   auto t1 = std::chrono::high_resolution_clock::now();
    //   double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //
    // Repeat with a much larger launch (e.g. <<<1024, 256>>>) and compare `ms`.
    // (Precise device-side timing with cudaEvents comes in a later day.)

    return 0;
}
