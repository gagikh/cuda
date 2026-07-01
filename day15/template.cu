// Day 15: Stream-Ordered Memory Allocation
// Goal: replace cudaMalloc/cudaFree with stream-ordered cudaMallocAsync/cudaFreeAsync,
// applied to a real image-contrast kernel, then benchmark allocation overhead.
//
// Compile:  nvcc -arch=sm_60 day15_template.cu -o day15 `pkg-config --cflags --libs opencv4`
// Run:      ./day15 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// TODO 1: contrast adjustment kernel: out = clamp((in - 128) * contrast + 128, 0, 255).
// `in`/`out` are flat (non-pitched) buffers -- allocated ourselves with
// cudaMallocAsync below, not through a GpuMat, since the point today is the
// allocator, not pitch handling.
__global__ void adjust_contrast(const unsigned char *in, unsigned char *out, int n, float contrast)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        // TODO: float v = (in[id] - 128.0f) * contrast + 128.0f;
        //       out[id] = (unsigned char)min(255.0f, max(0.0f, v));
    }
}

void run_with_malloc_async(cudaStream_t stream, const unsigned char *h_in,
                            unsigned char *h_out, int n, float contrast)
{
    unsigned char *d_in = nullptr, *d_out = nullptr;

    // TODO 1: allocate d_in/d_out with cudaMallocAsync(&ptr, n, stream) instead of cudaMalloc.
    // TODO: cudaMemcpyAsync(d_in, h_in, n, cudaMemcpyHostToDevice, stream);
    // TODO: launch adjust_contrast<<<grid, block, 0, stream>>>(d_in, d_out, n, contrast);
    // TODO: cudaMemcpyAsync(h_out, d_out, n, cudaMemcpyDeviceToHost, stream);
    // TODO: free d_in/d_out with cudaFreeAsync(ptr, stream) instead of cudaFree.
}

void benchmark_alloc_overhead(int iterations, size_t bytes)
{
    // TODO 2 (self-learning #2): time `iterations` iterations of cudaMalloc+cudaFree
    // vs. cudaMallocAsync+cudaFreeAsync for a small `bytes` allocation, using
    // cudaEvents around each loop. Which one wins, and by how much?
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
    CV_Assert(h_img.isContinuous());
    const int n = h_img.rows * h_img.cols;

    unsigned char *h_out;
    cudaMallocHost(&h_out, n); // pinned, for a clean async copy-out (see Day 4/7)

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    run_with_malloc_async(stream, h_img.data, h_out, n, /*contrast=*/1.5f);
    cudaStreamSynchronize(stream);

    cv::Mat h_result(h_img.size(), h_img.type(), h_out);
    cv::imshow("input", h_img);
    cv::imshow("contrast adjusted (cudaMallocAsync)", h_result);
    cv::waitKey(0);

    benchmark_alloc_overhead(1000, 4096);

    // TODO (self-learning #3): create an explicit cudaMemPool_t with cudaDeviceGetDefaultMemPool
    // or cudaMemPoolCreate, and drive allocations on it from two different streams.

    cudaFreeHost(h_out);
    cudaStreamDestroy(stream);

    return 0;
}
