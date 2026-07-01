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

// RAII graph wrapper, same shape as the graph_t in the README's Code
// Walkthrough — genericized to a raw cudaStream_t instead of
// cv::cuda::Stream, so this file has no OpenCV dependency.
enum class graph_status_t { UNINITIALIZED, CAPTURING, GRAPH_CREATED };

struct graph_t
{
    graph_status_t m_status = graph_status_t::UNINITIALIZED;
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_instance = nullptr;

    graph_t() = default;

    ~graph_t()
    {
        if (m_instance) {
            cudaGraphExecDestroy(m_instance);
            m_instance = nullptr;
        }
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
    }

    // non-copyable (owns GPU resources), movable if you need it later
    graph_t(const graph_t&) = delete;
    graph_t& operator=(const graph_t&) = delete;

    bool is_created() const { return graph_status_t::GRAPH_CREATED == m_status; }

    // TODO A: begin stream capture on `stream` (cudaStreamCaptureModeGlobal)
    // and set m_status to CAPTURING. Every op launched on `stream` after this
    // call is recorded into the graph instead of actually executing.
    void start_capture(cudaStream_t stream)
    {
        // TODO: cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // TODO: m_status = graph_status_t::CAPTURING;
    }

    // TODO B: end capture into m_graph, then instantiate m_instance from it.
    // Set m_status to GRAPH_CREATED when done.
    void create_graph(cudaStream_t stream)
    {
        // TODO: cudaStreamEndCapture(stream, &m_graph);
        // TODO: cudaGraphInstantiate(&m_instance, m_graph, nullptr, nullptr, 0);
        // TODO: m_status = graph_status_t::GRAPH_CREATED;
    }

    // TODO C: replay the captured graph on `stream`.
    void launch(cudaStream_t stream)
    {
        // TODO: cudaGraphLaunch(m_instance, stream);
    }
};

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
    graph_t graph;
    graph.start_capture(stream);
    transpose_shared<<<grid, block, 0, stream>>>(d_in, d_out, n);
    // TODO: add more ops here (e.g. a second kernel) to make graph capture worthwhile
    graph.create_graph(stream);

    // --- Launch the captured graph many times and time it ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; ++i) {
        graph.launch(stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("graph: %d launches in %.3f ms (%.4f ms/launch)\n", iterations, ms, ms / iterations);

    // TODO (self-learning #4): repeat the same `iterations` loop launching
    // transpose_shared directly (no graph) and compare per-launch overhead.

    // graph's destructor cleans up m_instance/m_graph automatically — no
    // manual cudaGraphExecDestroy/cudaGraphDestroy needed here.
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
