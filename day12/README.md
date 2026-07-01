# Day 12: CUDA Graph API

## Objectives
- Capture a sequence of kernel/memory operations into a CUDA graph
- Instantiate and launch a captured graph
- Compare matrix transpose via shared memory vs. via texture binding
- Understand when graph launch overhead pays off vs. regular sequential launches

## Key Concepts
- Graph recording
- Kernel + memory op capture
- Graph launch

## Resources
https://www.olcf.ornl.gov/wp-content/uploads/2021/10/013_CUDA_Graphs.pdf
https://developer.nvidia.com/blog/cuda-graphs/
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

Manual creation reference:
https://github.com/hummingtree/cuda-graph-with-dynamic-parameters/tree/release

## Code Walkthrough
A small RAII wrapper around a captured CUDA graph:

```c++
struct graph_t
{
    graph_creation_status_t m_status = UNINITIALIZED;
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_instance = nullptr;

    ~graph_t()
    {
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
        if (m_instance) {
            cudaGraphExecDestroy(m_instance);
            m_instance = nullptr;
        }
    }

    inline void launch(cv::cuda::Stream &stream)
    {
        CV_Assert(nullptr != m_instance);
        cudaGraphLaunch(m_instance, cv::cuda::custream(stream));
    }

    inline bool is_created() const { return GRAPH_CRATED == m_status; }
    inline bool is_initiaized() const { return INITIALIZED == m_status; }
    inline bool is_uninitiaized() const { return UNINITIALIZED == m_status; }
    inline void set_initiaized() { m_status = INITIALIZED; }

    static inline void start_capture(cv::cuda::Stream &stream) {
        cudaStreamBeginCapture(cv::cuda::custream(stream), cudaStreamCaptureModeGlobal);
    }

    inline 