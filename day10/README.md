# Day 10: Practical Algorithms

## Objectives
- Implement matrix multiplication on the GPU, naive then tiled
- Implement Hamming-distance descriptor matching
- Compare naive vs. optimized implementations

## Key Concepts
- Descriptor matching based on Hamming distance
- Matrix multiplication

## Resources
https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

## Hands-On Task
- Descriptor matching based on Hamming distance
- Matrix multiplication

## Self-Learning
1. Implement naive GPU matrix multiplication (global memory only).
2. Optimize it using shared-memory tiling (reuse Day 5 tiling patterns) and compare timing against the naive version.
3. Implement Hamming distance between binary descriptors using `__popc`.
4. Batch the Hamming distance computation to find the nearest descriptor match for each query descriptor.

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
