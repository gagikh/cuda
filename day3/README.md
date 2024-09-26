Warp execution and control flow

https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec3.pdf
https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

Hint: https://people.maths.ox.ac.uk/~gilesm/cuda/

- If/else
- for loop
- while/do while
- switch-case
+ loop unrolling

```c++
// --- Allocate temporary storage in shared memory 
#ifdef NDEBUG
	typedef cub::WarpReduce<int> WarpReduceT;

	 __shared__ typename WarpReduceT::TempStorage temp_storage;
	const auto result = WarpReduceT(temp_storage).Sum(r);
#else
	// in debug mode, it consumes much resources, so lets use this one
	int result = r;
#pragma unroll
	for (auto i = 1; i < 32; i *= 2) {
		result += __shfl_xor_sync(0xFFFFFFFF, result, i);
	}
#endif
```
![image](https://github.com/gagikh/cuda/assets/7694001/d483440c-3828-4ae7-8f7a-f6601242d0a5)


Task: large vector addition and time estimation, ignore warp logic at this moment

```c++
// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

// Allocate device memory for arrays d_A, d_B, and d_C
double *d_A, *d_B, *d_C;
cudaMalloc(&d_A, bytes);
cudaMalloc(&d_B, bytes);
cudaMalloc(&d_C, bytes);

// Copy data from host arrays A and B to device arrays d_A and d_B
cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

 // Launch kernel
add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

// Copy data from device array d_C to host array C
cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
```
