Nvidia cuda graph programming

https://www.olcf.ornl.gov/wp-content/uploads/2021/10/013_CUDA_Graphs.pdf
https://developer.nvidia.com/blog/cuda-graphs/


TODO:
Matrix transpose via shared memory, compare with the implementation using texture binding
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

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

    inline void create_graph(cv::cuda::Stream &stream)
    {
        CV_Assert(nullptr == m_instance);
        CV_Assert(nullptr == m_graph);
        CV_Assert(INITIALIZED == m_status);
        cudaStreamEndCapture(cv::cuda::custream(stream), &m_graph);
        cudaGraphInstantiate(&m_instance, m_graph, NULL, NULL, 0);
        m_status = GRAPH_CRATED;
    }
};

manual creation:
https://github.com/hummingtree/cuda-graph-with-dynamic-parameters/tree/release
