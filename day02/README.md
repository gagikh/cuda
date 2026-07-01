# Day 2: Thread Hierarchy & Execution Model

## Objectives
- Enumerate and divide work across threads, blocks, and grids
- Configure kernel launches correctly for 1D and 2D data
- Apply thread indexing patterns to real data (vectors, then images)
- Write a grid-stride loop that handles any input size with a fixed launch configuration

## Key Concepts
- Threads, blocks, grids: structure and enumeration
- Launch configuration and kernel invocation
- Thread indexing patterns
- Grid-stride loops

## Visual
![Grid made up of blocks, each block made up of a 2D array of threads, with the global-index formula shown](thread_hierarchy.svg)

A kernel launch creates a grid of blocks, and each block is itself a 1D/2D/3D array of threads. `blockIdx` tells a thread which block it's in; `threadIdx` tells it which slot within that block. The global-index formula in the diagram is the one pattern you'll reuse in nearly every kernel from here on.

## Grid-Stride Loops
Every kernel so far launches exactly enough threads to cover the data, one thread per element. That's fine for a toy example with a known, fixed `n` — but it breaks down fast:

- If `n` isn't known until runtime (loaded from a file, a network request, ...), you have to recompute the grid size every time.
- If `n` is huge, you may not be able to launch enough threads/blocks to cover it in one shot on every GPU.
- Get the grid-size arithmetic wrong (an off-by-one in `divUp`) and some elements are silently never processed — no error, no crash, just quietly wrong output.

The fix is a **grid-stride loop**: launch a *fixed* number of threads (chosen based on your GPU's occupancy, not on `n`), and have each thread process multiple elements in a loop, striding by the total thread count each time:

```c++
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
}
```

This is the pattern real (non-toy) CUDA code uses. `vector_add_grid_stride` in [`template.cu`](template.cu) is launched with a fixed `<<<256, 256>>>` regardless of `n` — try changing `n` to something much larger and confirm the same launch configuration still produces correct output.

## Resources
Threads, blocks, grids
- How to enumerate
- How to devide

[https://slideplayer.com/slide/15057888/](https://eximia.co/understanding-the-basics-of-cuda-thread-hierarchies/)

## Hands-On Task
Example project using VS — add 2 vectors (block/grid config, pipeline), then change it to add 2 images.

## Self-Learning
1. Implement 1D vector addition for a few different array sizes.
2. Extend the kernel to 2D thread indexing and add two grayscale images pixel-by-pixel.
3. Try block sizes of 32, 64, 128, and 256 threads and compare timing — get a first feel for occupancy.
4. Make the kernel correct for array sizes that are *not* an exact multiple of the block size (bounds checking).
5. Fill in `vector_add_grid_stride` in [`template.cu`](template.cu). Verify it produces the same result as `vector_add` for the current `n`, then bump `n` to something far larger than `blocks * threads` and confirm it's still correct without changing the launch configuration.

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
