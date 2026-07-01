# Day 3: Warp-Level Execution and Control Flow

## Objectives
- Define what a warp is and how it executes instructions in lockstep
- Reason about control flow (`if`/`else`, loops, `switch`) and its cost inside a warp
- Recognize and avoid warp divergence
- Apply loop unrolling where it helps

## Key Concepts
- Warp definition and behavior
- Control flow: `if`, `else`, `for`, `while`
- Loop unrolling
- Divergence impact and avoidance

## Resources
https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec3.pdf

https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

Hint: https://people.maths.ox.ac.uk/~gilesm/cuda/

- If/else
- for loop
- while/do while
- switch-case
+ loop unrolling

## Code Walkthrough
Warp-level reduction, with a debug-mode fallback using `__shfl_xor_sync`:

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

Large vector addition kernel (ignore warp logic for this part — see Self-Learning below):

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

## Hands-On Task
Large vector addition and time estimation (ignore warp logic at this point). Then: BGR to grayscale conversion with CUDA.

## Self-Learning
1. Implement the large vector addition above and time it against an equivalent CPU loop.
2. Convert a BGR image to grayscale in a kernel (`gray = 0.114*B + 0.587*G + 0.299*R`).
3. Deliberately introduce branch divergence (e.g. `if (threadIdx.x % 2 == 0)`) in a kernel and measure the performance hit vs. a divergence-free version.
4. Apply `#pragma unroll` to a small fixed-trip-count loop in one of your kernels and compare generated performance.

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
