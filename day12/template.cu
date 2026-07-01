// Day 12: CUDA Graph API
// Goal: shared-memory transpose of a real image, captured into a CUDA graph
// and launched repeatedly.
//
// Compile:  nvcc -arch=sm_50 day12_template.cu -o day12 `pkg-config --cflags --libs opencv4`
// Run:      ./day12 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

#define TILE_DIM 32

// TODO 1: shared-memory matrix transpose. Pad the tile ([TILE_DIM][TILE_DIM+1])
// to avoid bank conflicts on the write-back. `in`/`out` are GpuMat pointers;
// index rows via `in_step`/`out_step` (bytes), same pitched idea as Day 5+.
// Note `out` is height x width -> width x height (dimensions swap).
__global__ void transpose_shared(const unsigned char *in, size_t in_step,
                                  unsigned char *out, size_t out_step,
                                  int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM][TILE_DIM + 1]; // +1 padding avoids bank conflicts

    // TODO: load tile from `in` at (x, y) via in_step, __syncthreads(),
    //       write to `out` at the transposed location (x and y swapped)
    //       via out_step, using the padded tile.
}

// TODO 2 (self-learning #2): transpose_texture — same operation, but reading
// through a texture object bound to `in` (reuse Day 11's texture setup).

// RAII graph wrapper, same shape as the graph_t in the README's Code
// Walkthrough — genericized to a raw cudaStream_t instead of
// cv::cuda::Stream, so it works standalone without an OpenCV Stream.
enum class graph_status_t { UNINITIALIZED, CAPTURING, GRAPH_CREATED };

struct graph_t
{
    graph_status_t m_status = graph_status_t::UNINITIALIZED;
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_instance = nullptr;

    graph_t() = default;

    ~graph_t()
    {
        // Not CUDA_CHECK'd -- CUDA_CHECK calls exit() on failure, which is
        // not something a destructor should ever do (especially not during
        // stack unwinding from another exception). Report and move on.
        if (m_instance) {
            cudaError_t err = cudaGraphExecDestroy(m_instance);
            if (err != cudaSuccess) fprintf(stderr, "cudaGraphExecDestroy failed: %s\n", cudaGetErrorString(err));
            m_instance = nullptr;
        }
        if (m_graph) {
            cudaError_t err = cudaGraphDestroy(m_graph);
            if (err != cudaSuccess) fprintf(stderr, "cudaGraphDestroy failed: %s\n", cudaGetErrorString(err));
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
        // TODO: CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        // TODO: m_status = graph_status_t::CAPTURING;
    }

    // TODO B: end capture into m_graph, then instantiate m_instance from it.
    // Set m_status to GRAPH_CREATED when done.
    void create_graph(cudaStream_t stream)
    {
        // TODO: CUDA_CHECK(cudaStreamEndCapture(stream, &m_graph));
        // TODO: CUDA_CHECK(cudaGraphInstantiate(&m_instance, m_graph, nullptr, nullptr, 0));
        // TODO: m_status = graph_status_t::GRAPH_CREATED;
    }

    // TODO C: replay the captured graph on `stream`.
    void launch(cudaStream_t stream)
    {
        // TODO: CUDA_CHECK(cudaGraphLaunch(m_instance, stream));
    }
};

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

    cv::cuda::GpuMat d_in, d_out;
    d_in.upload(h_img);
    d_out.create(d_in.cols, d_in.rows, d_in.type()); // dimensions swap on transpose

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(cv::cudev::divUp(d_in.cols, TILE_DIM), cv::cudev::divUp(d_in.rows, TILE_DIM));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Capture a graph containing the transpose kernel ---
    graph_t graph;
    graph.start_capture(stream);
    transpose_shared<<<grid, block, 0, stream>>>(d_in.ptr<unsigned char>(), d_in.step,
                                                  d_out.ptr<unsigned char>(), d_out.step,
                                                  d_in.cols, d_in.rows);
    CUDA_CHECK_LAST_ERROR();
    // TODO: add more ops here (e.g. a second kernel) to make graph capture worthwhile
    graph.create_graph(stream);

    // --- Launch the captured graph many times and time it ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 1000;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        graph.launch(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("graph: %d launches in %.3f ms (%.4f ms/launch)\n", iterations, ms, ms / iterations);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cv::Mat h_out;
    d_out.download(h_out);
    cv::imshow("input", h_img);
    cv::imshow("transposed", h_out);
    cv::waitKey(0);

    // TODO (self-learning #4): repeat the same `iterations` loop launching
    // transpose_shared directly (no graph) and compare per-launch overhead.

    // graph's destructor cleans up m_instance/m_graph automatically — no
    // manual cudaGraphExecDestroy/cudaGraphDestroy needed here.
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
