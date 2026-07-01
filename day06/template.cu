// Day 6: Streams and Events
// Goal: image derivative kernel on a real image (via GpuMat), timed precisely with cudaEvents.
//
// Compile:  nvcc -arch=sm_50 day06_template.cu -o day06 `pkg-config --cflags --libs opencv4`
// Run:      ./day06 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

// TODO 1: image derivative kernel (simple central difference in x and y).
// dx[y][x] = img[y][x+1] - img[y][x-1]; dy similarly. Handle borders.
// `img`/`dx`/`dy` are raw GpuMat pointers; index rows via their respective
// `*_step` (bytes) -- same pitched-memory idea as Day 5.
__global__ void image_derivative(const unsigned char *img, size_t img_step,
                                  float *dx, float *dy, size_t grad_step,
                                  int width, int height)
{
    // TODO
}

// TODO 2 (self-learning #2): shared-memory convolution — reuse the Day 5 tiled_filter pattern.

// TODO 3 (self-learning #3): image transform kernel (rotate or scale).

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

    cv::cuda::GpuMat d_img, d_dx, d_dy;
    d_img.upload(h_img);
    d_dx.create(d_img.size(), CV_32F);
    d_dy.create(d_img.size(), CV_32F);

    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(d_img.cols, block.x), cv::cudev::divUp(d_img.rows, block.y));

    // First formal use of cudaEvent-based device-side timing.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    image_derivative<<<grid, block>>>(d_img.ptr<unsigned char>(), d_img.step,
                                       d_dx.ptr<float>(), d_dy.ptr<float>(), d_dx.step,
                                       d_img.cols, d_img.rows);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("image_derivative: %.3f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // TODO: dx/dy are raw float gradients -- cv::normalize (or take the
    // absolute value and scale to 0-255) before cv::imshow, or they'll
    // just look black/blown-out.
    cv::Mat h_dx;
    d_dx.download(h_dx);
    cv::imshow("input", h_img);
    cv::imshow("dx (raw float -- normalize me)", h_dx);
    cv::waitKey(0);

    // TODO (self-learning #5, stretch): create two cudaStream_t's and launch
    // independent work (e.g. derivative + transform) on each, then check for overlap.

    return 0;
}
