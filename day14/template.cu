// Day 14: CUDA Libraries
// Goal: estimate pi via Monte Carlo sampling using cuRAND, plus a bonus:
// fill a real cv::cuda::GpuMat with cuRAND-generated noise.
//
// Compile:  nvcc -arch=sm_50 day14_template.cu -o day14 -lcurand `pkg-config --cflags --libs opencv4`
// Run:      ./day14

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include "../common/cuda_check.h"

// TODO 1: initialize one curandState per thread, seeded uniquely per thread.
__global__ void setup_rng(curandState *states, unsigned long long seed, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n) return;
    // TODO: curand_init(seed, id, 0, &states[id]);
}

// TODO 2: each thread samples `samples_per_thread` random (x, y) points in [0,1)^2,
// counts how many fall inside the unit circle (x*x + y*y <= 1), and atomicAdds
// its count into a global counter.
__global__ void monte_carlo_pi(curandState *states, unsigned long long *inside_count,
                                int samples_per_thread, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n) return;

    // TODO: curandState local = states[id];
    //       int local_count = 0;
    //       for (int i = 0; i < samples_per_thread; ++i) {
    //           float x = curand_uniform(&local);
    //           float y = curand_uniform(&local);
    //           if (x*x + y*y <= 1.0f) ++local_count;
    //       }
    //       atomicAdd(inside_count, (unsigned long long)local_count);
}

// TODO (bonus): fill a GpuMat with random noise, one pixel per thread.
// `states` must be sized >= width*height (reuse setup_rng() to init it).
// `img`/`img_step` are the GpuMat's pointer/pitch, same as Day 5+.
__global__ void fill_noise_image(curandState *states, unsigned char *img, size_t img_step,
                                  int width, int height)
{
    // TODO: int x = ..., y = ...; if (x >= width || y >= height) return;
    //       int id = y * width + x; // index into `states`, NOT into `img` (states is flat)
    //       curandState local = states[id];
    //       img[y * img_step + x] = curand(&local) % 256;
}

int main()
{
    // --- Part 1: Monte Carlo pi estimation ---
    const int n = 1 << 16;
    const int samples_per_thread = 1000;

    curandState *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    unsigned long long *d_inside_count;
    CUDA_CHECK(cudaMalloc(&d_inside_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_inside_count, 0, sizeof(unsigned long long)));

    const int threads = 256;
    const int blocks = cv::cudev::divUp(n, threads);

    setup_rng<<<blocks, threads>>>(d_states, 1234ULL, n);
    CUDA_CHECK_LAST_ERROR();
    monte_carlo_pi<<<blocks, threads>>>(d_states, d_inside_count, samples_per_thread, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_inside_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_inside_count, d_inside_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double total_samples = (double)n * samples_per_thread;
    double pi_estimate = 4.0 * (double)h_inside_count / total_samples;
    printf("pi estimate: %f\n", pi_estimate);

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_inside_count));

    // TODO (self-learning #2/#3): add cuBLAS matrix-vector multiply and cuFFT examples here.

    // --- Part 2 (bonus): random noise image via GpuMat ---
    const int width = 256, height = 256;
    curandState *d_img_states;
    CUDA_CHECK(cudaMalloc(&d_img_states, width * height * sizeof(curandState)));
    setup_rng<<<cv::cudev::divUp(width * height, 256), 256>>>(d_img_states, 5678ULL, width * height);
    CUDA_CHECK_LAST_ERROR();

    cv::cuda::GpuMat d_noise(height, width, CV_8UC1);
    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(width, block.x), cv::cudev::divUp(height, block.y));
    fill_noise_image<<<grid, block>>>(d_img_states, d_noise.ptr<unsigned char>(), d_noise.step, width, height);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat h_noise;
    d_noise.download(h_noise);
    cv::imshow("cuRAND noise", h_noise);
    cv::waitKey(0);

    CUDA_CHECK(cudaFree(d_img_states));

    return 0;
}
