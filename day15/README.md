# Day 15: Stream-Ordered Memory Allocation

## Objectives
- Understand stream-ordered memory allocation semantics
- Use `cudaMallocAsync` / `cudaFreeAsync` correctly
- Use memory pools across multiple streams

## Key Concepts
- `cudaMallocAsync` / `cudaFreeAsync`
- Stream-ordered allocation semantics
- Memory pools

## Visual
![Classic cudaMalloc/cudaFree act as implicit device-wide sync points breaking stream concurrency, while cudaMallocAsync/cudaFreeAsync are ordered within a stream and reuse memory from a pool without a device-wide sync](stream_ordered_alloc.svg)

`cudaMalloc`/`cudaFree` are safe but blunt — they force the whole device to sync, which quietly kills the overlap you worked to set up in Day 6/7. The async versions are ordered within a single stream instead, so allocation composes with everything else: overlapping streams, and even capture into a CUDA graph (Day 12).

## Resources
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html

https://medium.com/@dmitrijtichonov/cuda-series-memory-and-allocation-fce29c965d37

## Hands-On Task
Replace a `cudaMalloc`/`cudaFree` pair with the stream-ordered `cudaMallocAsync`/`cudaFreeAsync` equivalents — applied to a real image-contrast kernel (loaded via `cv::imread`) instead of a synthetic vector add.

## Self-Learning
1. Fill in `run_with_malloc_async` in [`template.cu`](template.cu): allocate with `cudaMallocAsync`, copy the loaded image in, run `adjust_contrast`, copy the result out, and free with `cudaFreeAsync` — all on the same stream.
2. Benchmark allocation overhead: `cudaMalloc`/`cudaFree` vs. `cudaMallocAsync`/`cudaFreeAsync` in a loop of many small allocations.
3. Create an explicit `cudaMemPool_t` and use it across multiple streams; verify correctness with concurrent allocations.
4. Combine stream-ordered allocation with the Day 12 CUDA graph capture — capture allocation, kernel, and free into one graph.

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. Why does classic `cudaMalloc`/`cudaFree` act as an implicit device-wide synchronization point?
2. "Stream-ordered" is in the name `cudaMallocAsync` — ordered relative to what, specifically?
3. Why would you combine stream-ordered allocation with a CUDA graph (Day 12) instead of just using one or the other?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
