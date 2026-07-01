// Day 7: Asynchronous Execution Techniques
// Goal: overlap async H2D copy with kernel execution using streams.
//
// Compile:  nvcc -arch=sm_50 day7_template.cu -o day7
// Run:      ./day7

#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 1 << 22;
constexpr int NUM_CHUNKS = 4;
constexpr int CHUNK_SIZE = N / NUM_CHUNKS;

__global__ void process_chunk(const float *in, float *out, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) out[id] = in[id] * 2.0f; // placeholder work
}

int main()
{
    // Pinned host memory is required for true async cudaMemcpyAsync overlap (see Day 4).
    float *h_in, *h_out;
    cudaMallocHost(&h_in, N * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    // TODO: fill h_in with test data

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaStream_t streams[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // TODO: for each chunk i:
    //   1. cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes, H2D, streams[i])
    //   2. process_chunk<<<grid, block, 0, streams[i]>>>(...)
    //   3. cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes, D2H, streams[i])
    // Because each step is issued on its own stream, chunk i+1's copy can overlap
    // with chunk i's kernel — that's the overlap you're checking for.

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("chunked async pipeline: %.3f ms\n", ms);

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
