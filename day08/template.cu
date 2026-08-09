// Day 8: Warp-Level Intrinsics - Reduction
// Goal: warp-level sum reduction using __shfl_down_sync, then extract the
// indices of pixels above a threshold in a real image using warp scan.
//
// Compile:  nvcc -arch=sm_75 template.cu -o day08 `pkg-config --cflags --libs opencv4`
// Run:      ./day08 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

// TODO 1: warp-level sum reduction. 5 steps, halving the offset each time,
// after which LANE 0 holds the warp's total. (Lane 16 holds a 16-element
// partial, not the total -- shfl_down always moves data toward lower lanes.)
//
// No bounds check is needed on the source lane: when lane 20 asks for lane 36,
// the intrinsic returns lane 20's own value. Those lanes are past the useful
// half of the reduction and their results are discarded, so it never matters.
__device__ int warp_reduce_sum(int val)
{
    // TODO: for (int offset = 16; offset > 0; offset >>= 1)
    //           val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce_kernel(const int *in, int *out, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Note the shape of this line -- it is NOT `if (id < n) { ... }`.
    // Every lane must reach warp_reduce_sum, because the 0xFFFFFFFF mask
    // inside it promises all 32 will. Out-of-range lanes contribute the
    // identity for the operation instead (0 for sum). Wrapping the reduction
    // in a bounds check is undefined behaviour, not just a wrong total.
    int val = (id < n) ? in[id] : 0;

    val = warp_reduce_sum(val);

    // TODO: lane 0 of each warp writes its partial sum somewhere (shared mem or
    // atomicAdd to a global accumulator).
}

// TODO 4 (self-learning #4): reduce a whole BLOCK, not just a warp.
// A 256-thread block is 8 warps; warp_reduce_sum leaves you 8 partials that
// still need combining. The idiomatic pattern is hierarchical -- reduce inside
// each warp, park one value per warp in shared memory, then let the first warp
// reduce those. Two levels covers the 1024-thread maximum exactly, because
// 32 lanes collapse to 1 and at most 32 warps collapse to 1.
//
// Costs ONE __syncthreads() and 128 bytes of shared memory. The classic
// shared-memory tree reduction costs 8 barriers and 8 rounds of shared traffic
// for the same 256 threads -- time both (self-learning #4) and see.
__device__ int block_reduce_sum(int val)
{
    __shared__ int warp_sums[32];        // 32 warps max per block

    // Valid only because blockDim.x is a multiple of 32 here. For a block like
    // dim3(16,16) you must flatten the thread index first -- see INTRINSICS.md.
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    // TODO: val = warp_reduce_sum(val);
    // TODO: if (lane == 0) warp_sums[wid] = val;
    // TODO: __syncthreads();
    // TODO: const int nwarps = blockDim.x >> 5;
    //       val = (threadIdx.x < nwarps) ? warp_sums[lane] : 0;
    //       if (wid == 0) val = warp_reduce_sum(val);
    // returns the block total in thread 0
    (void)lane; (void)wid;
    return val;
}

// TODO 5 (self-learning #5): the butterfly variant. Identical cost, but every
// lane ends up holding the total instead of just lane 0 -- no broadcast needed
// afterwards. Useful when all 32 lanes need the result to continue.
//   for (int offset = 16; offset > 0; offset >>= 1)
//       val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
__device__ int warp_reduce_sum_all(int val)
{
    // TODO
    return val;
}

// TODO 2 (self-learning #2): warp-level inclusive prefix sum (scan).
// Kogge-Stone, 5 steps like the reduction. [1,1,1,...] -> [1,2,3,...].
//
// The `lane >= offset` guard is load-bearing here, unlike in the reduction.
// A lane asking for a source below 0 gets its OWN value back, and adding that
// would silently double its contribution. In warp_reduce_sum those lanes'
// results get discarded so the garbage never escapes; here they don't.
__device__ int warp_scan_inclusive(int val)
{
    // TODO: const int lane = threadIdx.x & 31;    // only valid if blockDim.x % 32 == 0
    //       for (int offset = 1; offset < 32; offset <<= 1) {
    //           const int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    //           if (lane >= offset) val += n;
    //       }
    return val;
}

// TODO 3 (self-learning #3, Hands-On Task): use warp_scan_inclusive to
// compact the indices of pixels above `threshold` into `out_indices`, and
// atomically bump `out_count` by each warp's local count. `img`/`img_step`
// are a GpuMat's raw pointer/pitch, same as Day 5-7.
__global__ void extract_indices_above_threshold(const unsigned char *img, size_t img_step,
                                                  int width, int height, unsigned char threshold,
                                                  int *out_indices, int *out_count)
{
    // TODO
}

// TODO 6 (self-learning #6): the same compaction, but for a BINARY predicate
// there's a two-instruction shortcut that beats the 5-step scan entirely.
// This is what production code does.
//
//   const int  lane = threadIdx.x & 31;
//   const bool keep = (pixel > threshold);
//
//   const unsigned ballot = __ballot_sync(0xFFFFFFFF, keep);  // 1 bit per lane
//   const int prefix = __popc(ballot & ((1u << lane) - 1));   // lanes before me
//   const int total  = __popc(ballot);                        // lanes in total
//
//   int base;
//   if (lane == 0) base = atomicAdd(out_count, total);        // ONE atomic per warp
//   base = __shfl_sync(0xFFFFFFFF, base, 0);                  // broadcast it
//
//   if (keep) out_indices[base + prefix] = my_index;
//
// The mask (1u << lane) - 1 clears bit `lane` and above, giving an EXCLUSIVE
// prefix -- your own slot shouldn't be counted before you write to it. It's
// correct at lane 31 too: (1u << 31) - 1 == 0x7FFFFFFF.
//
// One atomicAdd per warp instead of one per passing pixel: up to 32x less
// contention. Same warp-aggregation idea Day 9 generalizes as privatization.
__global__ void extract_indices_ballot(const unsigned char *img, size_t img_step,
                                        int width, int height, unsigned char threshold,
                                        int *out_indices, int *out_count)
{
    // TODO
}

int main(int argc, char **argv)
{
    // --- Part 1: generic warm-up, sum of 1s should equal n ---
    const int n = 1024;
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));

    // TODO: fill d_in with test data (e.g. all 1s to verify the sum == n)

    reduce_kernel<<<cv::cudev::divUp(n, 256), 256>>>(d_in, d_out, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out = 0;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    printf("sum = %d\n", h_out);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // --- Part 2: threshold + compact indices on a real image ---
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

    const int max_indices = h_img.rows * h_img.cols;
    int *d_indices, *d_count;
    CUDA_CHECK(cudaMalloc(&d_indices, max_indices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    dim3 block(32, 8); // multiple of warp size on x for clean warp_scan_inclusive use
    dim3 grid(cv::cudev::divUp(d_img.cols, block.x), cv::cudev::divUp(d_img.rows, block.y));
    extract_indices_above_threshold<<<grid, block>>>(
        d_img.ptr<unsigned char>(), d_img.step, d_img.cols, d_img.rows,
        128, d_indices, d_count);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("pixels above threshold: %d / %d\n", h_count, max_indices);

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_count));

    return 0;
}
