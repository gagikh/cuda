// examples/matrix_add.cu
//
// Worked reference for PITCHED 2D memory: cudaMallocPitch + cudaMemcpy2D.
// Referenced from Day 4 (allocation strategies) and Day 5, where the same
// pitch idea reappears as cv::cuda::GpuMat::step.
//
// The one idea to take away: a 2D device allocation is NOT width*height bytes
// laid out contiguously. cudaMallocPitch rounds each row up to a hardware-
// friendly alignment and hands you back that row stride ("pitch") in bytes.
// Every kernel touching the buffer must index rows by pitch, never by width.
//
// Compile:  nvcc -arch=sm_75 matrix_add.cu -o matrix_add
// Run:      ./matrix_add

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_check.h"

// TASK: measure the performance difference against host computation.
// TASK: enlarge N/M until the GPU actually wins, and explain where the
//       crossover is (hint: it's dominated by the two cudaMemcpy2D legs,
//       not by the arithmetic -- Day 1's host/device picture).
constexpr int N = 32;
constexpr int M = 32;

// Note `rows`/`cols` rather than N/M as parameter names: shadowing the file-
// scope constants with same-named parameters is legal but makes it hard to
// see which one a given line is using.
__global__ void device_add(const float *A, const float *B, float *C,
                           int rows, int cols,
                           size_t apitch, size_t bpitch, size_t cpitch)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;   // column
    const int y = threadIdx.y + blockIdx.y * blockDim.y;   // row
    if (y >= rows || x >= cols) {
        return;
    }

    // Row base = byte pointer + y * pitch. Cast to char* first so the
    // arithmetic is in bytes; only then reinterpret as float*.
    const float *aptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + y * apitch) + x;
    const float *bptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + y * bpitch) + x;
    float       *cptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + y * cpitch) + x;
    *cptr = *aptr + *bptr;
}

void host_add(const float hA[N][M], const float hB[N][M], float hC[N][M])
{
    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    // cudaMallocPitch writes the chosen row stride back into the pitch args.
    // It's normally larger than M * sizeof(float) -- print them and see.
    size_t dapitch = 0, dbpitch = 0, dcpitch = 0;
    CUDA_CHECK(cudaMallocPitch(&dA, &dapitch, M * sizeof(float), N));
    CUDA_CHECK(cudaMallocPitch(&dB, &dbpitch, M * sizeof(float), N));
    CUDA_CHECK(cudaMallocPitch(&dC, &dcpitch, M * sizeof(float), N));

    printf("requested row width: %zu bytes -> actual device pitch: %zu bytes\n",
           M * sizeof(float), dapitch);

    // cudaMemcpy2D(dst, dstPitch, src, srcPitch, widthInBytes, height, kind).
    // The host arrays are plain C arrays, so their "pitch" is exactly the row
    // width -- the whole point is that the two sides may differ.
    CUDA_CHECK(cudaMemcpy2D(dA, dapitch, hA, M * sizeof(float),
                            M * sizeof(float), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dB, dbpitch, hB, M * sizeof(float),
                            M * sizeof(float), N, cudaMemcpyHostToDevice));

    // TASK: extend this to a 3D grid (see cudaMalloc3D / cudaMemcpy3D).
    dim3 block(16, 16, 1);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              1);

    device_add<<<grid, block>>>(dA, dB, dC, N, M, dapitch, dbpitch, dcpitch);
    CUDA_CHECK_LAST_ERROR();   // <<<>>> returns nothing -- ask explicitly (Day 1)

    // cudaMemcpy2D is synchronous on the default stream and is ordered after
    // the kernel, so it doubles as the wait: no cudaDeviceSynchronize needed.
    CUDA_CHECK(cudaMemcpy2D(hC, M * sizeof(float), dC, dcpitch,
                            M * sizeof(float), N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    static float A[N][M];
    static float B[N][M];
    static float C[N][M];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = static_cast<float>(i + j);
            B[i][j] = static_cast<float>(i - j);
        }
    }

    host_add(A, B, C);

    // A[i][j] + B[i][j] == (i+j) + (i-j) == 2i, so every row should be
    // constant. Verifying against a known answer beats eyeballing 1024 numbers.
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (C[i][j] != static_cast<float>(2 * i)) ++mismatches;
        }
    }
    printf("%s (%d mismatches out of %d)\n",
           mismatches == 0 ? "OK" : "FAILED", mismatches, N * M);

    return mismatches == 0 ? 0 : 1;
}
