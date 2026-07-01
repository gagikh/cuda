// Day 8: Warp-Level Intrinsics - Reduction
// Goal: warp-level sum reduction using __shfl_down_sync.
//
// Compile:  nvcc -arch=sm_50 day8_template.cu -o day8
// Run:      ./day8

#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: warp-level sum reduction.
// Each of the 32 lanes contributes `val`; after this function every lane
// (or at least lane 0) holds the warp's total sum.
__device__ int warp_reduce_sum(int val)
{
    // TODO: for (int offset = 16; offset > 0; offset /= 2)
    //           val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce_kernel(const int *in, int *out, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int val = (id < n) ? in[id] : 0;

    val = warp_reduce_sum(val);

    // TODO: lane 0 of each warp writes its partial sum somewhere (shared mem or
    // atomicAdd to a global accumulator).
}

// TODO 2 (self-learning #2): warp-level inclusive prefix sum (scan).
__device__ int warp_scan_inclusive(int val)
{
    // TODO: for (int offset = 1; offset < 32; offset *= 2) {
    //           int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    //           if ((threadIdx.x & 31) >= offset) val += n;
    //       }
    return val;
}

// TODO 3 (self-learning #3): use warp_scan_inclusive to compact indices of
// values above a threshold into a contiguous output array.

int main()
{
    const int n = 1024;
    int *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    // TODO: fill d_in with test data (e.g. all 1s to verify the sum == n)

    reduce_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    int h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum = %d\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
