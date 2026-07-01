// Day 1: CUDA Basics and Programming Model
// Goal: compile, launch, and run a minimal kernel; inspect thread/block identity.
//
// Compile:  nvcc -arch=sm_50 day01_template.cu -o day01
// Run:      ./day01

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include "../common/cuda_check.h"
#include "../common/device_info.h"

// TODO 1: write a kernel that prints its own block/thread index using device-side printf.
__global__ void hello_kernel()
{
    // TODO: printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// TODO 2: each thread writes its own *raw* blockIdx.x / threadIdx.x — no combined
// "global index" formula (that's a Day 2 topic). Index each array directly by the
// raw field: one write per block into block_ids, one write per thread into thread_ids.
__global__ void identify_kernel(int *block_ids, int *thread_ids)
{
    // TODO: if (threadIdx.x == 0) block_ids[blockIdx.x] = blockIdx.x;
    // TODO: if (blockIdx.x == 0) thread_ids[threadIdx.x] = threadIdx.x;
}

int main()
{
    // --- Part 0: know your GPU ---
    // Run this first, always. Every "max threads per block", "shared mem per
    // SM", "warp size" number this course refers to abstractly is a concrete
    // number for YOUR GPU, printed right here.
    report_device_capabilities();

    // --- Part 1: say hello ---
    hello_kernel<<<2, 4>>>();
    // Kernel launches are asynchronous and return void -- the <<<>>> syntax
    // itself can't report a launch-configuration error. CUDA_CHECK_LAST_ERROR()
    // asks the runtime "did the last thing I launched fail?" immediately,
    // instead of the failure surfacing several lines later at an unrelated
    // CUDA_CHECK call (or not surfacing at all).
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Part 2: record + verify raw block/thread indices ---
    const int num_blocks = 4, threads_per_block = 4;
    int *d_block_ids = nullptr, *d_thread_ids = nullptr;
    int h_block_ids[num_blocks] = {0};
    int h_thread_ids[threads_per_block] = {0};

    CUDA_CHECK(cudaMalloc(&d_block_ids, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_thread_ids, threads_per_block * sizeof(int)));

    identify_kernel<<<num_blocks, threads_per_block>>>(d_block_ids, d_thread_ids);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaMemcpy(h_block_ids, d_block_ids, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thread_ids, d_thread_ids, threads_per_block * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_blocks; ++i) {
        printf("block_ids[%d] = %d\n", i, h_block_ids[i]);
    }
    for (int i = 0; i < threads_per_block; ++i) {
        printf("thread_ids[%d] = %d\n", i, h_thread_ids[i]);
    }

    CUDA_CHECK(cudaFree(d_block_ids));
    CUDA_CHECK(cudaFree(d_thread_ids));

    // TODO (self-learning #4): time a 1-thread launch vs a many-thread launch
    // from the host side using <chrono>, e.g.:
    //
    //   auto t0 = std::chrono::high_resolution_clock::now();
    //   hello_kernel<<<1, 1>>>();
    //   CUDA_CHECK_LAST_ERROR();
    //   CUDA_CHECK(cudaDeviceSynchronize());
    //   auto t1 = std::chrono::high_resolution_clock::now();
    //   double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //
    // Repeat with a much larger launch (e.g. <<<1024, 256>>>) and compare `ms`.
    // (Precise device-side timing with cudaEvents comes in a later day.)

    // TODO (self-learning #7): deliberately launch identify_kernel with an
    // invalid config, e.g. <<<1, 5000>>> (over the 1024 threads/block limit
    // on most GPUs). Run it once WITHOUT CUDA_CHECK_LAST_ERROR (notice
    // nothing visibly happens) and once WITH it (notice it now fails loudly,
    // right where the problem is).

    return 0;
}
