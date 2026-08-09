// Day 15: Stream-Ordered Memory Allocation
// Goal: replace cudaMalloc/cudaFree with stream-ordered cudaMallocAsync/cudaFreeAsync,
// applied to a real image-contrast kernel, then benchmark allocation overhead.
//
// Compile:  nvcc -arch=sm_75 template.cu -o day15 `pkg-config --cflags --libs opencv4`
//           (cudaMallocAsync needs compute capability >= 6.0 and CUDA >= 11.2;
//            sm_75 is this course's floor anyway.)
// Run:      ./day15 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "../common/cuda_check.h"

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

    // TODO 1: allocate d_in/d_out with CUDA_CHECK(cudaMallocAsync(&ptr, n, stream))
    //         instead of cudaMalloc.
    // TODO: CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, n, cudaMemcpyHostToDevice, stream));
    // TODO: launch adjust_contrast<<<grid, block, 0, stream>>>(d_in, d_out, n, contrast);
    //       CUDA_CHECK_LAST_ERROR();
    // TODO: CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, n, cudaMemcpyDeviceToHost, stream));
    // TODO: free d_in/d_out with CUDA_CHECK(cudaFreeAsync(ptr, stream)) instead of cudaFree.
}

void benchmark_alloc_overhead(int iterations, size_t bytes)
{
    // TODO 2 (self-learning #2): time `iterations` iterations of cudaMalloc+cudaFree
    // vs. cudaMallocAsync+cudaFreeAsync for a small `bytes` allocation, using
    // cudaEvents around each loop (CUDA_CHECK every call). Which one wins, and by how much?
    //
    // Then run the async loop a second time with the pool's release threshold
    // raised, and compare all three. By default the pool returns memory to the
    // OS at every device sync, which throws away the reuse you're paying for:
    //
    //   cudaMemPool_t pool;
    //   CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
    //   size_t threshold = 256ull * 1024 * 1024;   // or UINT64_MAX for "never"
    //   CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

// TODO 3 (self-learning #5): report what the pool is holding.
//   - cudaMemPoolAttrReservedMemCurrent = bytes the pool has taken from the OS
//   - cudaMemPoolAttrUsedMemCurrent     = bytes actually handed out right now
// A large gap means the pool is hoarding. cudaMemPoolTrimTo(pool, 0) gives it back.
void report_pool_usage(const char *label)
{
    // TODO: cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0);
    //       cuuint64_t reserved = 0, used = 0;
    //       cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    //       cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    //       printf("%-12s reserved %.2f MB, used %.2f MB\n", label,
    //              reserved / 1048576.0, used / 1048576.0);
    (void)label;
}

// TODO 4 (self-learning #3): build an explicit pool and allocate from it.
// Why not just use the default pool: isolation between subsystems, per-pool
// release policies, and IPC (set props.handleTypes to share across processes).
//
//   cudaMemPoolProps props = {};
//   props.allocType     = cudaMemAllocationTypePinned;
//   props.location.type = cudaMemLocationTypeDevice;
//   props.location.id   = 0;
//   // props.handleTypes = cudaMemHandleTypePosixFileDescriptor;  // IPC-capable
//   CUDA_CHECK(cudaMemPoolCreate(&pool, &props));
//   CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, bytes, pool, stream));
//   CUDA_CHECK(cudaFreeAsync(ptr, stream));      // no pool argument needed
//   CUDA_CHECK(cudaMemPoolDestroy(pool));        // after a sync
//
// A pool is per-DEVICE, not per-stream: drive the same pool from two streams
// and confirm both get valid, non-overlapping allocations.
void run_with_explicit_pool(cudaStream_t s1, cudaStream_t s2, size_t bytes)
{
    (void)s1; (void)s2; (void)bytes;
    // TODO
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
    CUDA_CHECK(cudaMallocHost(&h_out, n)); // pinned, for a clean async copy-out (see Day 4/7)

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    run_with_malloc_async(stream, h_img.data, h_out, n, /*contrast=*/1.5f);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cv::Mat h_result(h_img.size(), h_img.type(), h_out);
    cv::imshow("input", h_img);
    cv::imshow("contrast adjusted (cudaMallocAsync)", h_result);
    cv::waitKey(0);

    report_pool_usage("before");
    benchmark_alloc_overhead(1000, 4096);
    report_pool_usage("after");
    // TODO: cudaMemPoolTrimTo(pool, 0) here, then report again -- reserved should drop.

    // TODO (self-learning #3): drive an explicit pool from two streams.
    //   run_with_explicit_pool(stream, stream2, 1 << 20);

    // TODO (self-learning #6): the cross-stream bug worth making on purpose.
    // Allocate on `stream`, then launch adjust_contrast on a SECOND stream using
    // that pointer, with no event between them. Stream-ordered means ordered
    // *within one stream* -- another stream has no dependency on the allocation
    // and may run before it completes. It will often appear to work. Run it
    // under compute-sanitizer (Day 1), then fix it with a cudaEvent (Day 6).

    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
