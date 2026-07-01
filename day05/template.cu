// Day 5: Memory Conflicts and Shared Memory
// Goal: shared-memory tiled 2D filter, operating on a real image loaded via OpenCV.
//
// This is the first day template that takes an OpenCV dependency. From here
// on, day templates load real images/video through OpenCV and operate on
// cv::cuda::GpuMat instead of a plain device pointer you fill by hand.
//
// Compile:  nvcc -arch=sm_50 day05_template.cu -o day05 `pkg-config --cflags --libs opencv4`
// Run:      ./day05 <path-to-image>

#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#define TILE_DIM 16
#define RADIUS 1

// TODO: tiled 2D filter kernel (box blur, or a simple weighted average).
// `in`/`out` are raw pointers into GpuMat data; `in_step`/`out_step` are the
// GpuMat's row pitch in bytes -- GpuMat rows are not guaranteed contiguous,
// same idea as cudaMallocPitch (see examples/matrix_add.cu). Index rows by
// step, not by width.
__global__ void tiled_filter(const unsigned char *in, size_t in_step,
                              unsigned char *out, size_t out_step,
                              int width, int height)
{
    __shared__ unsigned char tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];

    // TODO: compute global x, y; load into `tile` (including the RADIUS halo),
    // reading each row via `in + row * in_step`, bounds-checked against
    // width/height (clamp at the image border).
    // TODO: __syncthreads();
    // TODO: compute the filtered value from the `tile` neighborhood and
    // write it to `out + y * out_step + x`, bounds-checked.
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <path-to-image>\n", argv[0]);
        return 1;
    }

    // --- Load a real image and upload it to the GPU ---
    cv::Mat h_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (h_img.empty()) {
        printf("failed to load image: %s\n", argv[1]);
        return 1;
    }

    cv::cuda::GpuMat d_in, d_out;
    d_in.upload(h_img);
    d_out.create(d_in.size(), d_in.type());

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((d_in.cols + TILE_DIM - 1) / TILE_DIM, (d_in.rows + TILE_DIM - 1) / TILE_DIM);

    tiled_filter<<<grid, block>>>(d_in.ptr<unsigned char>(), d_in.step,
                                   d_out.ptr<unsigned char>(), d_out.step,
                                   d_in.cols, d_in.rows);
    cudaDeviceSynchronize();

    // --- Download and display the result ---
    cv::Mat h_out;
    d_out.download(h_out);
    cv::imshow("input", h_img);
    cv::imshow("filtered", h_out);
    cv::waitKey(0);

    // TODO (self-learning #4, stretch): swap cv::imread + the single-shot
    // display above for a cv::VideoCapture loop -- upload each frame, run
    // the (by then Sobel) filter, cv::imshow the result, cv::waitKey(1),
    // and stop when the user presses a key or the stream ends.

    return 0;
}
