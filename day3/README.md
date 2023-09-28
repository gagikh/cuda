Warp execution and control flow

https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_03.pdf

- If/else
- for loop
- while/do while
- switch-case
+ loop unrolling

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

![image](https://github.com/gagikh/cuda/assets/7694001/d483440c-3828-4ae7-8f7a-f6601242d0a5)
