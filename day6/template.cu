// Day 6: Streams and Events
// Goal: image derivative kernel, timed precisely with cudaEvents.
//
// Compile:  nvcc -arch=sm_50 day6_template.cu -o day6
// Run:      ./day6

#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: image derivative kernel (simple central difference in x and y).
// dx[y][x] = img[y][x+1] - img[y][x-1]; dy similarly. Handle borders.
__global__ void image_derivative(const unsigned char *img, float *dx, float *dy,
                                  int width, int height)
{
    // TODO
}

// TODO 2 (self-learning #2): shared-memory convolution — reuse the Day 5 tiled_filter pattern.

// TODO 3 (self-learning #3): image transform kernel (rotate or scale).

int main()
{
    const int width = 256, height = 256;
    size_t img_bytes = width * height * sizeof(unsigned char);
    size_t out_bytes = width * height * sizeof(float);

    unsigned char *d_img;
    float *d_dx, *d_dy;
    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_dx, out_bytes);
    cudaMalloc(&d_dy, out_bytes);

    // TODO: fill d_img with test data

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // First formal use of cudaEvent-based device-side timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    image_derivative<<<grid, block>>>(d_img, d_dx, d_dy, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("image_derivative: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_img);
    cudaFree(d_dx);
    cudaFree(d_dy);

    // TODO (self-learning #5, stretch): create two cudaStream_t's and launch
    // independent work (e.g. derivative + transform) on each, then check for overlap.

    return 0;
}
