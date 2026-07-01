// Day 7: Asynchronous Execution Techniques
// Goal: overlap async H2D copy with kernel execution using streams, applied
// to horizontal bands of a real image loaded via OpenCV.
//
// Compile:  nvcc -arch=sm_50 day07_template.cu -o day07 `pkg-config --cflags --libs opencv4`
// Run:      ./day07 <path-to-image>

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudev.hpp>

constexpr int NUM_CHUNKS = 4;

// Placeholder work: brighten each pixel. Swap for something real once
// you've worked through the examination-practice tasks below.
__global__ void process_chunk(const unsigned char *in, unsigned char *out, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) out[id] = min(255, in[id] + 40);
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
    CV_Assert(h_img.isContinuous()); // keeps row-chunking == flat byte-chunking below

    const int total = h_img.rows * h_img.cols;
    const int rows_per_chunk = cv::cudev::divUp(h_img.rows, NUM_CHUNKS);
    const int chunk_elems = rows_per_chunk * h_img.cols;

    // Pinned host memory is required for true async cudaMemcpyAsync overlap (see Day 4).
    unsigned char *h_in, *h_out;
    cudaMallocHost(&h_in, total);
    cudaMallocHost(&h_out, total);
    memcpy(h_in, h_img.data, total);

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, total);
    cudaMalloc(&d_out, total);

    cudaStream_t streams[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // TODO: for each chunk i (bounds-check the last chunk against `total`,
    // since rows_per_chunk * NUM_CHUNKS may overshoot h_img.rows):
    //   1. cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes, H2D, streams[i])
    //   2. process_chunk<<<grid, block, 0, streams[i]>>>(d_in + offset, d_out + offset, chunk_elems)
    //   3. cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes, D2H, streams[i])
    // Because each step is issued on its own stream, chunk i+1's copy can overlap
    // with chunk i's kernel -- that's the overlap you're checking for.

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("chunked async pipeline: %.3f ms\n", ms);

    // Wrap the pinned output buffer as a cv::Mat header (no copy) to display it.
    cv::Mat h_result(h_img.size(), h_img.type(), h_out);
    cv::imshow("input", h_img);
    cv::imshow("brightened (chunked async)", h_result);
    cv::waitKey(0);

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
