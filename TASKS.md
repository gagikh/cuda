# 100 CUDA Practice Tasks

A consolidated problem bank: 5 tasks per day (75 total, drawn from each day's Self-Learning section) plus 25 bonus tasks that go beyond the 15-day structure. Use this as a single place to pick a task to practice, independent of which day you're currently on — each task is tagged with the day whose material it depends on, or `(Bonus)` if it isn't covered by any specific day.

Nothing here has an answer key. See [GLOSSARY.md](GLOSSARY.md) if a term is unfamiliar, and the relevant `dayNN/README.md` for background before attempting that day's tasks.

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

## Day 9 — Warp-Level Data Exchange
41. Compute an image's mean pixel value using warp reduction + `atomicAdd`. *(Day 9)*
42. Pack 32 binary pixel values into one 32-bit word using `__ballot_sync`. *(Day 9)*
43. Write the inverse "unzip" operation. *(Day 9)*
44. Implement `pyrDown` (blur + downsample by 2). *(Day 9)*
45. Implement `pyrUp` (upsample by 2 + blur). *(Day 9)*

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
77. Implement histogram equalization on the GPU.
78. Implement a Canny edge detector from scratch (gradient → non-max suppression → hysteresis).
79. Implement bilateral filtering (edge-preserving blur).
80. Implement a median filter using a small sorting network in shared memory.
81. Implement image thresholding — both fixed and adaptive — as a kernel.
82. Implement a simple optical flow estimator (Lucas-Kanade, small window).
83. Implement alpha blending of two images on the GPU.
84. Implement an RGB-to-HSV color-space conversion kernel.
85. Implement a perspective warp (homography) kernel using textures.
86. Implement non-maximum suppression for corner detection.
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
