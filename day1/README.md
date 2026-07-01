# Day 1: CUDA Basics and Programming Model

## Objectives
- Explain the host/device relationship and why GPUs parallelize work differently than CPUs
- Describe the CUDA programming model: kernels, threads, blocks, grids
- Compile and run a first `.cu` program with `nvcc`
- Reason about GPU architecture fundamentals (SMs, cores, warps at a high level)

## Key Concepts
- CUDA programming model overview
- Host vs Device
- GPU architecture fundamentals
- Thread hierarchy overview

## Resources
Lecture:
https://harmanani.github.io/classes/csc447/Notes/Lecture02.pdf

// Programming model
https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

// PPT
http://developer.download.nvidia.com/compute/developertrainingmaterials/presentations/cuda_language/Introduction_to_CUDA_C.pptx

// Resources
https://developer.nvidia.com/cuda-education

https://harmanani.github.io/csc447.html

https://ww2.cs.fsu.edu/~guidry/cuda.ppt

## Hands-On Task
Write and run a minimal kernel that identifies itself (block/thread index) from the device, launched with a small grid/block configuration. This is the "hello world" of CUDA — the goal is a clean compile/run cycle, not performance.

## Self-Learning
Small tasks to reinforce today's material, roughly in increasing difficulty:

1. Write a kernel that prints `Hello from block X, thread Y` using device-side `printf`.
2. Launch the same kernel with different grid/block configurations (e.g. `<<<1,1>>>`, `<<<2,4>>>`, `<<<4,32>>>`) and observe how the identifiers change.
3. Write a kernel that computes each thread's *global* index (`blockDim.x * blockIdx.x + threadIdx.x`) and writes it into an output array; copy it back and verify on the host.
4. Launch a trivial kernel with 1 thread vs. with thousands of threads and time both from the host using `<chrono>` (wrap the launch + `cudaDeviceSynchronize()`) — this is your first look at why parallelism matters. Precise device-side timing (`cudaEvent`s) is covered later.

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
