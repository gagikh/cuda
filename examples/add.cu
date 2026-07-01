#include <cuda.h>
#include <iostream>

// TODO: 2D addition is not perfect here,
// TASK: measure the performance diff against host computation

// enlarge the sizes
constexpr auto N = 32;
constexpr auto M = 32;

__global__ void device_add(float *A, float *B, float *C, size_t N, size_t M, size_t apitch, size_t bpitch, size_t cpitch)
{
    const auto x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= N || x >= M) {
	    return;
    }

    const auto aptr = reinterpret_cast<float*>(((char*)A) + y * apitch + x * sizeof(float));
    const auto bptr = reinterpret_cast<float*>(((char*)B) + y * bpitch + x * sizeof(float));
    const auto cptr = reinterpret_cast<float*>(((char*)C) + y * cpitch + x * sizeof(float));
    *cptr = *aptr + *bptr;
}

void host_add(float hA[N][M], float hB[N][M], float hC[N][M])
{
    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    size_t dapitch, dbpitch, dcpitch;
    cudaMallocPitch(&dA, &dapitch, M * sizeof(float), N);
    cudaMallocPitch(&dB, &dbpitch, M * sizeof(float), N);
    cudaMallocPitch(&dC, &dcpitch, M * sizeof(float), N);

    cudaMemcpy2D (dA, dapitch, hA, M * sizeof(float), M * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D (dB, dbpitch, hB, M * sizeof(float), M * sizeof(float), N,	cudaMemcpyHostToDevice);

    dim3 block;

    // TODO: task for 3D complex multiplication
    dim3 grid;
    block.x = 32;
    block.y = 32;
    block.z = 1;

    
    grid.x = (M + block.x - 1) / block.x;
    grid.y = (N + block.y - 1) / block.y;
    grid.z = 1;

    // invokde kernel
    device_add<<<grid, block>>>(dA, dB, dC, N, M, dapitch, dbpitch, dcpitch);
    cudaMemcpy2D (hC, M * sizeof(float), dC, dcpitch, M * sizeof(float), N,	cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

// compile: nvcc -ccbin gcc-7 -gencode arch=compute_50,code=sm_50 add.cu -lcudart -lstdc++;
int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);

    float A[N][M];
    float B[N][M];
    float C[N][M];

    for (auto i = 0; i < N; ++i) {
	    for (auto j = 0; j < M; ++j) {
	      A[i][j] = i + j;
	      B[i][j] = i - j;
	    }
    }
    host_add(A, B, C);
    for (auto i = 0; i < N; ++i) {
	    for (auto j = 0; j < M; ++j) {
	      std::cout << C[i][j] << ", ";
	    }
	    std::cout << std::endl;
    }
    return 0;
}
