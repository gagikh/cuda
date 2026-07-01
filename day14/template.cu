// Day 14: CUDA Libraries
// Goal: estimate pi via Monte Carlo sampling using cuRAND.
//
// Compile:  nvcc -arch=sm_50 day14_template.cu -o day14 -lcurand
// Run:      ./day14

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

int main()
{
    const int n = 1 << 16;
    const int samples_per_thread = 1000;

    curandState *d_states;
    cudaMalloc(&d_states, n * sizeof(curandState));

    unsigned long long *d_inside_count;
    cudaMalloc(&d_inside_count, sizeof(unsigned long long));
    cudaMemset(d_inside_count, 0, sizeof(unsigned long long));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    setup_rng<<<blocks, threads>>>(d_states, 1234ULL, n);
    monte_carlo_pi<<<blocks, threads>>>(d_states, d_inside_count, samples_per_thread, n);
    cudaDeviceSynchronize();

    unsigned long long h_inside_count = 0;
    cudaMemcpy(&h_inside_count, d_inside_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double total_samples = (double)n * samples_per_thread;
    double pi_estimate = 4.0 * (double)h_inside_count / total_samples;
    printf("pi estimate: %f\n", pi_estimate);

    cudaFree(d_states);
    cudaFree(d_inside_count);

    // TODO (self-learning #2/#3): add cuBLAS matrix-vector multiply and cuFFT examples here.

    return 0;
}
