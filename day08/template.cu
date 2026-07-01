// Day 8: Warp-Level Intrinsics - Reduction
// Goal: warp-level sum reduction using __shfl_down_sync, then extract the
// indices of pixels above a threshold in a real image using warp scan.
//
// Compile:  nvcc -arch=sm_50 day08_template.cu -o day08 `pkg-config --cflags --libs opencv4`
// Run:      ./day08 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

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

// TODO 3 (self-learning #3, Hands-On Task): use warp_scan_inclusive to
// compact the indices of pixels above `threshold` into `out_indices`, and
// atomically bump `out_count` by each warp's local count. `img`/`img_step`
// are a GpuMat's raw pointer/pitch, same as Day 5-7.
__global__ void extract_indices_above_threshold(const unsigned char *img, size_t img_step,
                                                  int width, int height, unsigned char threshold,
                                                  int *out_indices, int *out_count)
{
    // TODO
}

int main(int argc, char **argv)
{
    // --- Part 1: generic warm-up, sum of 1s should equal n ---
    const int n = 1024;
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));

    // TODO: fill d_in with test data (e.g. all 1s to verify the sum == n)

    reduce_kernel<<<cv::cudev::divUp(n, 256), 256>>>(d_in, d_out, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out = 0;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    printf("sum = %d\n", h_out);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // --- Part 2: threshold + compact indices on a real image ---
    if (argc < 2) {
        printf("(skipping Part 2: usage: %s <path-to-image>)\n", argv[0]);
        return 0;
    }

    cv::Mat h_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (h_img.empty()) {
        printf("failed to load image: %s\n", argv[1]);
        return 1;
    }

    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);

    const int max_indices = h_img.rows * h_img.cols;
    int *d_indices, *d_count;
    CUDA_CHECK(cudaMalloc(&d_indices, max_indices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    dim3 block(32, 8); // multiple of warp size on x for clean warp_scan_inclusive use
    dim3 grid(cv::cudev::divUp(d_img.cols, block.x), cv::cudev::divUp(d_img.rows, block.y));
    extract_indices_above_threshold<<<grid, block>>>(
        d_img.ptr<unsigned char>(), d_img.step, d_img.cols, d_img.rows,
        128, d_indices, d_count);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("pixels above threshold: %d / %d\n", h_count, max_indices);

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_count));

    return 0;
}
