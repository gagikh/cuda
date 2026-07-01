# Day 8: Warp-Level Intrinsics – Reduction

## Objectives
- Understand warp shuffle functions and intra-warp communication
- Implement warp-level parallel reduction
- Reason about performance tuning at the warp level

## Key Concepts
- Warp shuffle functions
- Intra-warp communication
- Parallel reduction
- Performance tuning

## Resources
https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf

https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf

https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

## Hands-On Task
32-order FFT; extract indices of an image above a threshold (use scan).

## Self-Learning
1. Implement warp-level sum reduction using `__shfl_down_sync`.
2. Implement a simple parallel prefix sum (scan) within a single warp.
3. Use the scan result to extract (compact) the indices of pixels above a threshold.
4. (Stretch) Implement a 32-point FFT butterfly using warp shuffles.

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
