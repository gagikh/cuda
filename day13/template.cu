// Day 13: Cache Behavior and Optimization
// Goal: apply __ldg and bank-conflict-free shared memory to a real image.
//
// Compile:  nvcc -arch=sm_50 day13_template.cu -o day13 `pkg-config --cflags --libs opencv4`
// Run:      ./day13 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>

#define TILE_DIM 16
#define RADIUS 1

// Baseline: the Day 5 tiled filter, unmodified. `in`/`out` are GpuMat
// pointers; `in_step`/`out_step` are their row pitch in bytes (Day 5+).
__global__ void tiled_filter_baseline(const unsigned char *in, size_t in_step,
                                       unsigned char *out, size_t out_step,
                                       int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
    // TODO: same body as Day 5's tiled_filter, but index rows via in_step/out_step
}

// TODO 1 (self-learning #1): same filter, but read `in` through __ldg() since it's
// read-only for the duration of the kernel.
__global__ void tiled_filter_ldg(const unsigned char *__restrict__ in, size_t in_step,
                                  unsigned char *out, size_t out_step,
                                  int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
    // TODO: load into `tile` using __ldg(&in[row * in_step + col]) instead of
    // direct indexing
}

// TODO 2 (self-learning #2): apply col ^ row swizzling instead of padding to
// remove bank conflicts (see the Visual section / swizzling.svg in the README).
//
// The idea: index shared memory as tile[r][c ^ r] everywhere -- both when
// writing into shared memory and when reading neighbors back out. XOR is
// its own inverse, so using the same formula both times keeps the logical
// layout correct; only the physical bank each element lands on changes.
// No extra padding column needed, unlike tiled_filter_baseline.
//
// Caveat to work through: the clean permutation property of col ^ row only
// holds when the row width is a power of two (so it lines up with the
// 32-bank layout). TILE_DIM + 2*RADIUS here is 18, not a power of two --
// decide whether to swizzle only the inner TILE_DIM-wide (power-of-two)
// region, or round the shared-memory row width up to the next power of two
// and mask the swizzled index.
__global__ void tiled_filter_swizzled(const unsigned char *in, size_t in_step,
                                       unsigned char *out, size_t out_step,
                                       int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
    // TODO: same load/compute as tiled_filter_baseline, but replace every
    // tile[r][c] access with tile[r][c ^ r].
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

    cv::cuda::GpuMat d_in, d_out;
    d_in.upload(h_img);
    d_out.create(d_in.size(), d_in.type());

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(cv::cudev::divUp(d_in.cols, TILE_DIM), cv::cudev::divUp(d_in.rows, TILE_DIM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_filter_baseline<<<grid, block>>>(d_in.ptr<unsigned char>(), d_in.step,
                                            d_out.ptr<unsigned char>(), d_out.step,
                                            d_in.cols, d_in.rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_baseline = 0.0f;
    cudaEventElapsedTime(&ms_baseline, start, stop);
    printf("baseline: %.3f ms\n", ms_baseline);

    cudaEventRecord(start);
    tiled_filter_ldg<<<grid, block>>>(d_in.ptr<unsigned char>(), d_in.step,
                                       d_out.ptr<unsigned char>(), d_out.step,
                                       d_in.cols, d_in.rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_ldg = 0.0f;
    cudaEventElapsedTime(&ms_ldg, start, stop);
    printf("__ldg:    %.3f ms\n", ms_ldg);

    cudaEventRecord(start);
    tiled_filter_swizzled<<<grid, block>>>(d_in.ptr<unsigned char>(), d_in.step,
                                            d_out.ptr<unsigned char>(), d_out.step,
                                            d_in.cols, d_in.rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_swizzled = 0.0f;
    cudaEventElapsedTime(&ms_swizzled, start, stop);
    printf("swizzled: %.3f ms\n", ms_swizzled);

    cv::Mat h_out;
    d_out.download(h_out);
    cv::imshow("input", h_img);
    cv::imshow("filtered", h_out);
    cv::waitKey(0);

    return 0;
}
