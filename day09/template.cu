// Day 9: Warp-Level Data Exchange
// Goal: image mean via warp reduction + atomicAdd, and __ballot_sync bit packing,
// on a real image loaded via OpenCV.
//
// Compile:  nvcc -arch=sm_50 day09_template.cu -o day09 `pkg-config --cflags --libs opencv4`
// Run:      ./day09 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

// TODO 1: compute image mean.
// Each thread reads one pixel (indexing rows via `img_step`, the GpuMat
// pitch -- same idea as Day 5-8), warp-reduces the sum (as in Day 8), then
// lane 0 of each warp does atomicAdd into a single global accumulator.
__global__ void image_sum_kernel(const unsigned char *img, size_t img_step,
                                  int width, int height, unsigned long long *total)
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

// TODO (self-learning #3/#4): pyrDown / pyrUp kernels, operating on GpuMat
// data/step like image_sum_kernel above.
__global__ void pyr_down_kernel(const unsigned char *in, size_t in_step,
                                 unsigned char *out, size_t out_step,
                                 int width, int height)
{
    // TODO: 5x5 Gaussian blur + sample every other pixel
}

__global__ void pyr_up_kernel(const unsigned char *in, size_t in_step,
                               unsigned char *out, size_t out_step,
                               int width, int height)
{
    // TODO: insert zeros + 5x5 Gaussian blur (scaled by 4)
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <path-to-image>\n", argv[0]);
        return 1;
    }

    cv::Mat h_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (h_img.empty()) {
        printf("failed to load image: %s\n", argv[1]);
        return 1;
    }

    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);

    unsigned long long *d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));

    dim3 block(32, 8); // multiple of warp size on x, matches warp-reduction assumptions
    dim3 grid(cv::cudev::divUp(d_img.cols, block.x), cv::cudev::divUp(d_img.rows, block.y));
    image_sum_kernel<<<grid, block>>>(d_img.ptr<unsigned char>(), d_img.step,
                                       d_img.cols, d_img.rows, d_total);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_total = 0;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    printf("mean = %f\n", static_cast<double>(h_total) / (d_img.cols * d_img.rows));

    CUDA_CHECK(cudaFree(d_total));

    // TODO (self-learning #3/#4): allocate a half-size GpuMat, run pyr_down_kernel,
    // cv::imshow the result next to the original.

    return 0;
}
