# Day 5: Memory Conflicts and Shared Memory

## Objectives
- Understand shared memory banking and how bank conflicts happen
- Correctly synchronize reads/writes to shared memory (`__syncthreads()`)
- Distinguish global, shared, constant, and pitched memory and when to use each
- Implement a shared-memory tiled filter

## Key Concepts
- Bank conflicts
- Sync read/write in kernel
- Global memory
- Shared memory
- Constant memory
- Pitched memory

## Resources
http://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec35.html

Task reference: https://developer.download.nvidia.com/compute/DevZone/C/html_x64/3_Imaging/convolutionSeparable/doc/convolutionSeparable.pdf

## Reference Implementation
[`examples/matrix_add.cu`](../examples/matrix_add.cu) at the repo root uses `cudaMallocPitch` / `cudaMemcpy2D` — a working example of pitched memory referenced in this day's material.

## Hands-On Task
Use shared memory for a 2D filter. Final task: 2D Sobel filter implementation on a video stream.

## Self-Learning
1. Implement a shared-memory tile-based 2D convolution filter (start with a simple box blur).
2. Deliberately create a shared-memory access pattern with bank conflicts, measure the perf hit, then fix it with padding.
3. Implement a 2D Sobel filter using shared memory.
4. Extend the Sobel filter to process a video stream frame by frame.

## Code Template
See [`template.cu`](template.cu) for a skeleton t