// common/cuda_check.h
//
// Shared error-checking macros used by every day's template.cu from Day 1
// onward. Include with a relative path, e.g. from dayNN/template.cu:
//
//   #include "../common/cuda_check.h"
//
// Why this exists: CUDA API calls (cudaMalloc, cudaMemcpy, ...) return a
// cudaError_t that is silently ignored if you don't check it. Kernel
// launches are worse -- the <<<...>>> syntax doesn't return anything at
// all, so a bad launch configuration (too many threads per block, an
// invalid grid size, ...) fails *silently* unless you explicitly ask the
// runtime "did the last thing I launched fail?" afterward.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap any CUDA runtime call that returns a cudaError_t:
//   CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
//   CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n  in call: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err__), #call);   \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Call right after a kernel launch, since <<<grid, block>>> itself has no
// return value to check. This catches launch-configuration errors
// (invalid grid/block dims, too much shared memory requested, etc.)
// immediately, instead of a mysterious failure several lines later on the
// next unrelated CUDA_CHECK call.
//
//   my_kernel<<<grid, block>>>(...);
//   CUDA_CHECK_LAST_ERROR();
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())
