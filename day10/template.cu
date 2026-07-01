// Day 10: Practical Algorithms
// Goal: naive matrix multiplication, then Hamming distance matching on real
// ORB descriptors extracted from an image.
//
// Compile:  nvcc -arch=sm_50 day10_template.cu -o day10 `pkg-config --cflags --libs opencv4`
// Run:      ./day10 <path-to-image>

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

#define TILE_DIM 16

// TODO 1: naive matrix multiplication. C = A * B, all N x N.
__global__ void matmul_naive(const float *A, const float *B, float *C, int n)
{
    // TODO: int row = ..., col = ...;
    //       float sum = 0; for (k in 0..n) sum += A[row*n+k] * B[k*n+col];
    //       C[row*n+col] = sum;
}

// TODO 2 (self-learning #2): shared-memory tiled matrix multiplication.
// Compare timing against matmul_naive for the same n.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // TODO: loop over tiles, load As/Bs, __syncthreads(), accumulate, __syncthreads()
}

// TODO 3 (self-learning #3): Hamming distance between two 32-bit descriptors using __popc.
__device__ int hamming_distance(unsigned int a, unsigned int b)
{
    // TODO: return __popc(a ^ b);
    return 0;
}

// TODO 4 (self-learning #4): for each query descriptor, find the index of the
// closest descriptor in a reference set (smallest Hamming distance).
//
// NOTE: real ORB descriptors are 256 bits (32 bytes) each; this simplified
// version only compares one 32-bit word per descriptor (see main() below).
// A real matcher would carry all 8 words per descriptor and sum
// hamming_distance() across them.
__global__ void match_descriptors(const unsigned int *queries, int num_queries,
                                   const unsigned int *refs, int num_refs,
                                   int *best_match_idx)
{
    // TODO
}

int main(int argc, char **argv)
{
    // --- Part 1: generic matmul warm-up ---
    const int n = 256;
    size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // TODO: fill d_A, d_B with test data

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(cv::cudev::divUp(n, TILE_DIM), cv::cudev::divUp(n, TILE_DIM));

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // --- Part 2: Hamming distance matching on real ORB descriptors ---
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

    auto orb = cv::cuda::ORB::create();
    cv::cuda::GpuMat d_descriptors;
    std::vector<cv::KeyPoint> keypoints;
    orb->detectAndCompute(d_img, cv::cuda::GpuMat(), keypoints, d_descriptors);

    printf("found %d ORB keypoints, %d bytes/descriptor\n",
           (int)keypoints.size(), d_descriptors.empty() ? 0 : d_descriptors.cols);

    if (d_descriptors.empty()) {
        printf("no descriptors found -- try a different image\n");
        return 0;
    }

    // Descriptor rows are pitched (GpuMat), so download to host and pull out
    // one dense, contiguous uint32 per descriptor -- keeps match_descriptors
    // focused on the Hamming-distance algorithm, not pitch handling.
    cv::Mat h_descriptors;
    d_descriptors.download(h_descriptors);

    std::vector<unsigned int> h_words(h_descriptors.rows);
    for (int i = 0; i < h_descriptors.rows; ++i) {
        h_words[i] = *reinterpret_cast<const unsigned int*>(h_descriptors.ptr<unsigned char>(i));
    }

    unsigned int *d_words;
    CUDA_CHECK(cudaMalloc(&d_words, h_words.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_words, h_words.data(), h_words.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    int *d_best_match;
    CUDA_CHECK(cudaMalloc(&d_best_match, h_words.size() * sizeof(int)));

    // Self-match as a sanity check: matching a descriptor set against itself
    // should give best_match_idx[i] == i with distance 0.
    const int num_desc = static_cast<int>(h_words.size());
    match_descriptors<<<cv::cudev::divUp(num_desc, 256), 256>>>(d_words, num_desc, d_words, num_desc, d_best_match);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_words));
    CUDA_CHECK(cudaFree(d_best_match));

    return 0;
}
