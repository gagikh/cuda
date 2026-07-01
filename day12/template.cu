// Day 12: CUDA Graph API
// Goal: shared-memory matrix transpose, captured into a CUDA graph and launched repeatedly.
//
// Compile:  nvcc -arch=sm_50 day12_template.cu -o day12
// Run:      ./day12

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 32

// TODO 1: shared-memory matrix transpose. Pad the tile ([TILE_DIM][TILE_DIM+1])
// to avoid bank conflicts on the write-back.
__global__ void transpose_shared(const float *in, float *out, int n)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 padding avoids bank conflicts

    // TODO: load tile from `in` at (x, y), __syncthreads(),
    //       write to `out` at the transposed location using the padded tile.
}

// TODO 2 (self-learning #2): transpose_texture — same operation, but reading
// through a texture object bound to `in` (reuse Day 11's texture setup).

int main()
{
    const int n = 512;
    size_t bytes = n * n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // TODO: fill d_in with test data

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(n / TILE_DIM, n / TILE_DIM);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Capture a graph containing the transpose kernel ---
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    transpose_shared<<<grid, block, 0, stream>>>(d_in, d_out, n);
    // TODO: add more ops here (e.g. a second kernel) to make graph capture worthwhile
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    // --- Launch the captured graph many times and time it ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; ++i) {
        cudaGraphLaunch(instance, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("graph: %d launches in %.3f ms (%.4f ms/launch)\n", iterations, ms, ms / iterations);

    // TODO (self-learning #4): repeat the same `iterations` loop launching
    // transpose_shared directly (no graph) and compare per-launch overhead.

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
