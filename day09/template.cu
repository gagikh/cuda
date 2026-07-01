// Day 9: Warp-Level Data Exchange
// Goal: image mean via warp reduction + atomicAdd, and __ballot_sync bit packing.
//
// Compile:  nvcc -arch=sm_50 day9_template.cu -o day9
// Run:      ./day9

#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: compute image mean.
// Each thread reads one pixel, warp-reduces the sum (as in Day 8), then lane 0
// of each warp does atomicAdd into a single global accumulator.
__global__ void image_sum_kernel(const unsigned char *img, unsigned long long *total, int n)
{
    // TODO
}

// TODO 2: zip 32 binary values (0/1) held one-per-lane into a single 32-bit word
// using __ballot_sync, written by lane 0.
__global__ void zip_binary_kernel(const unsigned char *bits, unsigned int *packed, int n)
{
    // TODO: unsigned int mask = __ballot_sync(0xFFFFFFFF, bits[id] != 0);
    //       if ((threadIdx.x & 31) == 0) packed[...] = mask;
}

// TODO: unzip_binary_kernel — inverse of zip_binary_kernel.
__global__ void unzip_binary_kernel(const unsigned int *packed, unsigned char *bits, int n)
{
    // TODO
}

// TODO (self-learning #3/#4): pyrDown / pyrUp kernels.
__global__ void pyr_down_kernel(const unsigned char *in, unsigned char *out,
                                 int width, int height)
{
    // TODO: 5x5 Gaussian blur + sample every other pixel
}

__global__ void pyr_up_kernel(const unsigned char *in, unsigned char *out,
                               int width, int height)
{
    // TODO: insert zeros + 5x5 Gaussian blur (scaled by 4)
}

int main()
{
    const int width = 256, height = 256;
    const int n = width * height;

    unsigned char *d_img;
    unsigned long long *d_total;
    cudaMalloc(&d_img, n);
    cudaMalloc(&d_total, sizeof(unsigned long long));
    cudaMemset(d_total, 0, sizeof(unsigned long long));

    // TODO: fill d_img with test data

    image_sum_kernel<<<(n + 255) / 256, 256>>>(d_img, d_total, n);
    cudaDeviceSynchronize();

    unsigned long long h_total = 0;
    cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("mean = %f\n", static_cast<double>(h_total) / n);

    cudaFree(d_img);
    cudaFree(d_total);

    return 0;
}
