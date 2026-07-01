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

## Visual
![Conflict-free shared memory access where each thread hits a different bank, versus a bank conflict where multiple threads hit bank 0 due to stride-32 access](bank_conflicts.svg)

Shared memory is split into 32 banks so that 32 threads can be serviced in one transaction — but only if each thread hits a different bank. Stride-32 access patterns (common when indexing by a tile width that's a multiple of 32) collapse onto the same bank and get serialized. Padding the row stride by one element is the standard fix, and it's exactly what `tiled_filter` in [`template.cu`](template.cu) is set up for.

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
See [`template.cu`](template.cu) for a skeleton to start from.
