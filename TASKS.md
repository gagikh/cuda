# 100 CUDA Practice Tasks

A consolidated problem bank: roughly 5 tasks per day, drawn from each day's Self-Learning section, plus 25 bonus tasks (76–100) that go beyond the 15-day structure. Lettered entries (`10a`, `45b`, ...) are later additions kept in place so the original numbering doesn't shift. Each task is tagged with the day whose material it depends on, or `(Bonus)` if no specific day covers it.

Nothing here has an answer key. See [GLOSSARY.md](GLOSSARY.md) if a term is unfamiliar, [PERFORMANCE.md](PERFORMANCE.md) for the optimization tasks, [INTRINSICS.md](INTRINSICS.md) for the device functions, and the relevant `dayNN/README.md` for background before attempting that day's tasks.

## Day 1 — CUDA Basics and Programming Model
1. Print block/thread identity from the device using `printf`, across several different launch configurations. *(Day 1)*
2. Write raw `blockIdx`/`threadIdx` values from the device into host-verifiable arrays. *(Day 1)*
3. Time a 1-thread launch vs. a many-thread launch with `<chrono>` and explain the difference. *(Day 1)*
4. Compile with `--keep` and read the generated `.ptx` file. *(Day 1)*
5. Run `report_device_capabilities()` and write down your GPU's warp size, max threads/block, and shared memory per SM. *(Day 1)*

## Day 2 — Thread Hierarchy & Execution Model
6. Implement 1D vector addition for a few different array sizes. *(Day 2)*
7. Extend to 2D thread indexing and add two grayscale images pixel-by-pixel. *(Day 2)*
8. Compare timing across block sizes 32, 64, 128, and 256. *(Day 2)*
9. Make a kernel correct for array sizes that aren't an exact multiple of the block size. *(Day 2)*
10. Implement a grid-stride loop and verify it's still correct at 10x the original `n`, with the same launch configuration. *(Day 2)*
10a. Write a copy kernel two ways — indexed `blockIdx.x * blockDim.x + threadIdx.x` vs. `threadIdx.x * gridDim.x + blockIdx.x` — and measure the ratio. Both are correct; predict the gap before running. *(Day 2)*
10b. Swap the row/column roles of `threadIdx.x` and `threadIdx.y` in a 2D image kernel and measure the slowdown. *(Day 2)*
10c. Call `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for block sizes 32/64/128/256 and check where measured timing stops tracking occupancy. *(Day 2)*

## Day 3 — Warp-Level Execution and Control Flow
11. Implement large vector addition and time it against an equivalent CPU loop. *(Day 3)*
12. Convert a BGR image to grayscale in a kernel. *(Day 3)*
13. Deliberately introduce branch divergence and measure the performance hit. *(Day 3)*
14. Apply `#pragma unroll` to a fixed-trip-count loop and compare generated performance. *(Day 3)*
15. Sketch the fetch/decode/register-read/execute/memory/writeback pipeline for 4 instructions across 6 cycles, on paper. *(Day 3)*

## Day 4 — CUDA Memory Types and Management
16. Benchmark `cudaMemcpy` with pageable vs. pinned host memory for a large transfer. *(Day 4)*
17. Page-lock an existing pageable buffer with `cudaHostRegister` instead of allocating pinned memory up front. *(Day 4)*
18. Rewrite vector-add to use `cudaMallocManaged` (unified memory). *(Day 4)*
19. Profile pageable/pinned/unified variants with Nsight Systems and compare the timelines. *(Day 4)*
20. Try mapped (zero-copy) memory and compare its transfer behavior to pinned. *(Day 4)*

## Day 5 — Memory Conflicts and Shared Memory
21. Implement a shared-memory tile-based 2D box blur. *(Day 5)*
22. Deliberately create a bank-conflicting access pattern, measure the hit, then fix it with padding. *(Day 5)*
23. Implement a 2D Sobel filter using shared memory. *(Day 5)*
24. Extend the Sobel filter to process a video stream frame by frame. *(Day 5)*
25. Load and display a real image with `cv::imread`/`cv::imshow` before writing any kernel logic. *(Day 5)*

## Day 6 — Streams and Events
26. Implement an image derivative (gradient) kernel — compute dx/dy per pixel. *(Day 6)*
27. Reuse the Day 5 tiling approach for a shared-memory convolution. *(Day 6)*
28. Implement a simple image transform (rotate or scale) kernel. *(Day 6)*
29. Time kernels precisely with `cudaEvent`s and compare against `<chrono>` measurements. *(Day 6)*
30. Split independent work across two CUDA streams and check whether they overlap. *(Day 6)*

## Day 7 — Asynchronous Execution Techniques
31. Overlap an async H2D copy with kernel execution using two streams and `cudaMemcpyAsync`. *(Day 7)*
32. Compute vector mean and standard deviation on the GPU, compare against a CPU implementation. *(Day 7)*
33. Implement image dilation and/or erosion filters. *(Day 7)*
34. Implement a small (32x32) matrix multiplication kernel. *(Day 7)*
35. Chunk a real image into horizontal bands and pipeline copy-in/compute/copy-out across streams. *(Day 7)*

## Day 8 — Warp-Level Intrinsics: Reduction
36. Implement warp-level sum reduction using `__shfl_down_sync`. *(Day 8)*
37. Implement an inclusive prefix sum (scan) within a single warp. *(Day 8)*
38. Use the scan result to compact indices of pixels above a threshold. *(Day 8)*
39. Implement a 32-point FFT butterfly using warp shuffles. *(Day 8)*
40. Compare warp-shuffle reduction against a shared-memory reduction for the same problem size. *(Bonus)*
40a. Extend the warp reduction to a full `block_reduce_sum` (warp → shared → warp), then reduce a whole grid via one `atomicAdd` per block. Count the `__syncthreads()` calls against a classic tree reduction. *(Day 8)*
40b. Rewrite the reduction with `__shfl_xor_sync` so every lane holds the total; confirm the cost is unchanged and say when you'd want it. *(Day 8)*
40c. Redo the compaction with `__ballot_sync` + `__popc` and one warp-aggregated atomic. Time it against the scan version at 5% and 95% pass rates. *(Day 8)*
40d. Write the divergence bug deliberately — `warp_reduce_sum` inside `if (id < n)` with `n` not a multiple of 32 — and run it under `compute-sanitizer --tool synccheck`. *(Day 8)*

## Day 9 — Warp-Level Data Exchange
41. Compute an image's mean pixel value using warp reduction + `atomicAdd`. *(Day 9)*
42. Pack 32 binary pixel values into one 32-bit word using `__ballot_sync`. *(Day 9)*
43. Write the inverse "unzip" operation. *(Day 9)*
44. Implement `pyrDown` (blur + downsample by 2). *(Day 9)*
45. Implement `pyrUp` (upsample by 2 + blur). *(Day 9)*
45a. Build a 256-bin histogram twice — naive global `atomicAdd` per pixel vs. privatized into shared memory — and measure the ratio on a normal image and on a near-uniform one. *(Day 9)*
45b. Replace the shared-memory `atomicAdd` in the privatized histogram with `atomicAdd_block` and measure the difference. *(Day 9)*

## Day 10 — Practical Algorithms
46. Implement naive GPU matrix multiplication. *(Day 10)*
47. Optimize it with shared-memory tiling and compare timing against the naive version. *(Day 10)*
48. Implement Hamming distance between binary descriptors using `__popc`. *(Day 10)*
49. Batch-match a query descriptor set against a reference set, finding each nearest neighbor. *(Day 10)*
50. Extract real ORB descriptors from an image and self-match them as a correctness sanity check. *(Day 10)*

## Day 11 — Textures and Surfaces
51. Build a CUDA texture object bound to a real image. *(Day 11)*
52. Implement image zoom (upscale) using `tex2D` bilinear filtering. *(Day 11)*
53. Implement image rotation via inverse-mapped texture sampling. *(Day 11)*
54. Compare texture-based zoom against a manual shared-memory bilinear implementation. *(Day 11)*
55. Explain, in your own words, why `cudaAddressModeClamp` changes the result specifically at image borders. *(Day 11)*

## Day 12 — CUDA Graph API
56. Implement matrix transpose using shared memory, padded to avoid bank conflicts. *(Day 12)*
57. Implement the same transpose using texture binding and compare performance. *(Day 12)*
58. Capture a multi-kernel pipeline into a CUDA graph via stream capture. *(Day 12)*
59. Launch the captured graph 1000 times and compare total time against 1000 direct launches. *(Day 12)*
60. Add a memory operation (not just a kernel) into the same captured graph. *(Bonus)*

## Day 13 — Cache Behavior and Optimization
61. Add `__ldg()` to a read-heavy kernel from an earlier day and measure the effect. *(Day 13)*
62. Implement `col ^ row` swizzling to remove bank conflicts without a padding column. *(Day 13)*
63. Experiment with L2 persistence hints (`cudaAccessPolicyWindow`) on a repeatedly-read buffer. *(Day 13)*
64. Optimize the Day 6 image transform kernel using every technique from the week so far. *(Day 13)*
65. Benchmark `__ldg`, swizzling, and padding on the same kernel and rank them for your GPU. *(Bonus)*
65a. Add achieved-bandwidth and %-of-peak output to all three timings in `day13/template.cu`, and decide from the numbers whether the day's optimizations ever had room to help. *(Day 13)*
65b. Coarsen `tiled_filter_baseline` to 2, 4 and 8 outputs per thread; plot time against elements-per-thread and correlate the drop-off with register spilling from `-Xptxas -v`. *(Day 13)*
65c. Sweep the grid size of the Day 2 grid-stride vector add (64 → 4096 blocks) at fixed `n` and explain the curve as coarsening at one end, occupancy at the other. *(Day 13)*

## Day 14 — CUDA Libraries
66. Estimate π via Monte Carlo sampling with cuRAND. *(Day 14)*
67. Use cuBLAS for a matrix-vector multiply and compare against your Day 10 kernel. *(Day 14)*
68. Use cuFFT to compute an FFT and compare against your Day 8 32-point attempt. *(Day 14)*
69. Fill a `GpuMat` with cuRAND-generated noise and display it. *(Day 14)*
70. Try a recursive/dynamic-parallelism kernel launch — have a kernel launch a child kernel. *(Day 14)*

## Day 15 — Stream-Ordered Memory Allocation
71. Replace a `cudaMalloc`/`cudaFree` pair with `cudaMallocAsync`/`cudaFreeAsync` on a stream. *(Day 15)*
72. Benchmark allocation overhead: classic vs. stream-ordered, over many small allocations. *(Day 15)*
73. Create an explicit `cudaMemPool_t` and drive allocations on it from two different streams. *(Day 15)*
74. Combine stream-ordered allocation with a Day 12 CUDA graph capture. *(Day 15)*
75. Apply `cudaMallocAsync` to a real image-processing kernel end to end. *(Day 15)*

## Bonus: Image Processing with OpenCV / GpuMat
76. Implement a Gaussian blur kernel and compare it to `cv::cuda::GaussianFilter`.
77. Implement histogram equalization on the GPU. (The histogram itself is exam task [E2](#e2--histogram-of-a-grayscale-image); equalization adds the scan and the lookup on top.)
78. Implement a Canny edge detector from scratch (gradient → non-max suppression → hysteresis). Full specification as exam task [E3](#e3--canny-edge-detection).
79. Implement bilateral filtering (edge-preserving blur).
80. Implement a median filter using a small sorting network in shared memory.
81. Implement image thresholding — both fixed and adaptive — as a kernel.
82. Implement a simple optical flow estimator (Lucas-Kanade, small window).
83. Implement alpha blending of two images on the GPU.
84. Implement an RGB-to-HSV color-space conversion kernel.
85. Implement a perspective warp (homography) kernel using textures.
86. Implement non-maximum suppression for corner detection.
86b. Implement connected components labeling of a binary mask. Full specification as exam task [E1](#e1--connected-components-labeling-of-a-binary-mask).
87. Build a real-time webcam filter pipeline: `cv::VideoCapture` → GPU kernel → `cv::imshow`.
88. Implement a full Laplacian pyramid blend of two images.
89. Implement a separable box filter (horizontal pass, then vertical) and compare it to a single 2D tiled pass.
90. Implement template matching (normalized cross-correlation) on the GPU.

## Bonus: Advanced / Beyond This Course
91. Implement a grid-wide reduction using cooperative groups (no host round-trip between blocks).
92. Split a vector-add workload across two GPUs with `cudaSetDevice(0)`/`cudaSetDevice(1)`.
93. Enable peer-to-peer memory access between two GPUs with `cudaDeviceEnablePeerAccess`, if you have more than one.
94. Profile a kernel with Nsight Compute and determine whether it's compute-bound or memory-bound.
95. Implement dynamic parallelism: a kernel that launches a child kernel based on data computed at runtime.
96. Implement a persistent-kernel pattern — a kernel that loops internally pulling work from a queue, instead of being relaunched per item.
97. Port one of your Day 5-13 kernels to cooperative groups' `tiled_partition` instead of raw warp intrinsics.
98. Implement one kernel in half precision (FP16) and compare accuracy and speed against FP32.
99. Build a small CUDA unit-test harness that compares kernel output against a CPU reference for randomized inputs.
100. Write a one-page performance report for any kernel from this course: measured throughput, theoretical peak from `report_device_capabilities()`, and the percentage of peak achieved.

---

# Final Exam

Three practical tasks. Each one is a real computer-vision algorithm, each draws on a different part of the course, and all three operate on a real image through `cv::cuda::GpuMat` the way every day from Day 5 onward has.

**What is assessed, for all three:**

1. **Correctness** — output matches an OpenCV reference on the same input. Not "looks about right": compare arrays and report how many pixels differ.
2. **Error handling** — every CUDA call through `CUDA_CHECK`, every launch followed by `CUDA_CHECK_LAST_ERROR()`, and the program runs clean under `compute-sanitizer` (Day 1).
3. **Measurement** — `cudaEvent` timing averaged over many runs ([`common/timer.h`](common/timer.h)), not a single `<chrono>` reading, and the result expressed as a percentage of your GPU's theoretical peak (Day 13, [PERFORMANCE.md](PERFORMANCE.md)).
4. **Explanation** — say *why* your kernel performs as it does, in the vocabulary of the course: coalescing, occupancy, bank conflicts, atomic contention, memory- vs compute-bound.

A correct kernel with no measurement is an incomplete answer. So is a fast kernel you cannot explain.

**Baselines must be honest.** Compare against OpenCV's GPU implementation (`cv::cuda::*`) where one exists, not against a naive single-threaded CPU loop. A "100× speedup" over unoptimized CPU code is not a result.

---

## E1 · Connected Components Labeling of a binary mask

Assign every connected region of a binary mask a unique label, so that two pixels share a label exactly when a path of set pixels joins them.

- **The question.** Where does the time actually go — the labeling passes, or the convergence checking? And how does the answer change as the number of components grows?
- **Input.** Any binary mask: threshold a grayscale image (Day 5), or generate synthetic blobs. Test both few-large-components and many-small-components cases, because they stress different things.
- **Minimum to pass.** A working label-propagation implementation (iterate: each pixel takes the minimum label of its 4- or 8-neighbourhood, repeat until nothing changes), correctness checked against `cv::connectedComponents`, and timing as a function of component count.
- **To go further.** Implement union-find with path compression as a second version and find the crossover. Do the convergence test on the device with a single `__device__` flag instead of copying a flag back every iteration.
- **Draws on.** Day 2 (2D indexing), Day 5 (shared-memory tiles with a halo — neighbours cross tile edges), Day 9 (atomics), Day 13 (coalescing).
- **Typical mistakes.** Copying a "did anything change?" flag to the host every iteration, so the measurement is dominated by round trips rather than by the algorithm. Forgetting that neighbour reads cross tile boundaries and need a halo. Treating an iteration count that varies with the image as if it were fixed.

## E2 · Histogram of a grayscale image

Count how many pixels fall in each of 256 intensity bins.

- **The question.** How much does atomic contention actually cost, and how much of it does privatization remove?
- **Input.** Any grayscale image. Test a normal photograph **and** a near-uniform one (mostly a single shade), because the second is where contention becomes visible.
- **Minimum to pass.** Two implementations — one global `atomicAdd` per pixel, one privatized into a shared-memory histogram and merged once per block (Day 9) — identical output, verified against `cv::calcHist`, with the speedup measured on both images and the difference explained.
- **To go further.** Add warp-aggregated atomics using `__ballot_sync` + `__popc` (Day 8). Count the global atomics each version issues and check that the measured speedup tracks that number.
- **Draws on.** Day 5 (shared memory), Day 8 (warp aggregation), Day 9 (atomics, privatization), Day 13 (% of peak — a histogram is memory-bound, so know what it is *allowed* to achieve).
- **Typical mistakes.** Timing the first launch, so the measurement includes context setup. Forgetting to zero the shared histogram, or to `__syncthreads()` after zeroing and again before merging. Reporting a speedup from only the easy image, where contention is low and privatization barely helps.

## E3 · Canny edge detection

The full pipeline: Gaussian blur → gradient magnitude and direction (Sobel) → non-maximum suppression → double threshold → hysteresis.

- **The question.** Which stage dominates, and is the whole pipeline limited by arithmetic or by the number of times you cross global memory?
- **Input.** Any grayscale image; a video stream via `cv::VideoCapture` for the stretch version.
- **Minimum to pass.** All five stages as CUDA kernels, output compared against `cv::Canny` with the same parameters (report the percentage of differing pixels — exact equality is not expected, since tie-breaking in suppression differs), and **per-stage timing** showing which stage costs what.
- **To go further.** Fuse adjacent stages to cut global-memory round trips and measure what fusing buys ([PERFORMANCE.md §7](PERFORMANCE.md)). Use a separable Gaussian instead of a 2D one. Capture the whole pipeline into a CUDA graph (Day 12) and compare per-frame launch overhead. Run it on video with the stages overlapped across streams (Day 7).
- **Draws on.** Day 5 (tiling with halo), Day 6 (events, per-stage timing), Day 7 (stream overlap for video), Day 9 (hysteresis propagation needs atomics or an iteration like E1), Day 12 (graphs), Day 13 (fusion, % of peak).
- **Typical mistakes.** Doing hysteresis on the host because the propagation is awkward, then reporting a GPU timing that hides a device-to-host round trip per frame. Timing only the kernels and not the transfers. Comparing a tuned GPU pipeline against a debug-build CPU baseline.

---

**Suggested deliverable for each task:** the source, a correctness report (how it was verified and how many pixels differed), a timing table, and half a page explaining the performance. Task 100 above is the template for that last part.
