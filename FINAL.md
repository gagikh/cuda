# Final Exam · Day 1 – Day 15

The final exam covers the whole course, Day 1 through Day 15. The midterm ([MIDTERM.md](MIDTERM.md)) covers Day 8 – Day 15 only.

A ticket has **two theory questions**, and the practical part is the **defence of the student’s independent assignment** given during the semester. The question list and the assignment list are published below in full.

**Which scale applies.** The two universities grade this course differently; the questions and the assignments are the same for both.

| University | Final exam | Midterm |
|---|---|---|
| **RAU** | 100 points, broken down in the table below | — |
| **YSU** | 9 units | 4 units — [MIDTERM.md](MIDTERM.md), Day 8 – Day 15 |

**Marking, RAU scale (100 points).**

| Part | Points | What is assessed |
|---|---|---|
| Theory question 1 | 30 | Accurate definitions (15), explanation in your own words with an example or a calculation (10), answers to follow-up questions (5) |
| Theory question 2 | 30 | Same |
| Defence of the independent assignment | 40 | Correctness against the reference (20), error handling with `CUDA_CHECK` (7), `cudaEvent` timing averaged over many runs and expressed as % of peak (8), explanation of the performance and answers on the code (5) |

A correct kernel that has not been measured does not get full marks, and neither does a fast one you cannot explain.

---

## Theory questions

<a id="q1"></a>**Q1.** The CUDA programming model: host and device, kernel, thread, block, grid. *([Day 1](day01/README.md))*

<a id="q2"></a>**Q2.** Error handling: `CUDA_CHECK`, why a kernel launch needs a separate check rather than a return value, and the difference between synchronous and asynchronous errors. *([Day 1](day01/README.md))*

<a id="q3"></a>**Q3.** The thread hierarchy, built-in variables, computing the global index in 1D and 2D. *([Day 2](day02/README.md))*

<a id="q4"></a>**Q4.** Coalesced global-memory access; the number of transactions for sequential and strided access. *([Day 2](day02/README.md))*

<a id="q5"></a>**Q5.** Occupancy and the choice of block size. *([Day 2](day02/README.md))*

<a id="q6"></a>**Q6.** The SIMT model, the warp, the instruction pipeline, latency hiding. *([Day 3](day03/README.md))*

<a id="q7"></a>**Q7.** Branch divergence and reconvergence; loop unrolling. *([Day 3](day03/README.md))*

<a id="q8"></a>**Q8.** Pageable, pinned, mapped and unified host memory: the five types compared, and when each is the right choice. *([Day 4](day04/README.md))*

<a id="q9"></a>**Q9.** What page-locking actually does: the virtual-to-physical mapping, why the DMA engine needs a stable physical address, and the cost of over-pinning. *([Day 4](day04/README.md))*

<a id="q10"></a>**Q10.** Zero-copy versus unified memory: per-access bus traffic against per-migration page traffic, dirty-page ping-ponging, and `__managed__`. *([Day 4](day04/README.md))*

<a id="q11"></a>**Q11.** Shared memory: banks, conflicts, broadcast, `__syncthreads()`. *([Day 5](day05/README.md))*

<a id="q12"></a>**Q12.** Removing bank conflicts: padding and XOR swizzling. *([Day 5](day05/README.md))*

<a id="q13"></a>**Q13.** Constant and pitched memory. *([Day 5](day05/README.md))*

<a id="q14"></a>**Q14.** CUDA streams and events; device-side timing. *([Day 6](day06/README.md))*

<a id="q15"></a>**Q15.** How the host thread waits for the GPU: the `cudaSetDeviceFlags` policies and why it must be the first CUDA call; host functions in a stream with `cudaLaunchHostFunc`. *([Day 6](day06/README.md))*

<a id="q16"></a>**Q16.** Asynchronous copies, double buffering, inter-stream dependencies. *([Day 7](day07/README.md))*

<a id="q17"></a>**Q17.** Pipelining a large image across streams: chunking into bands, overlapping copy-in / compute / copy-out, and why page-locked host memory is mandatory for it. *([Day 7](day07/README.md))*

<a id="q18"></a>**Q18.** Warp and lane; the lane mask; the shuffle functions and the `width` parameter. *([Day 8](day08/README.md))*

<a id="q19"></a>**Q19.** Warp- and block-level reduction; the warp → shared → warp scheme. *([Day 8](day08/README.md))*

<a id="q20"></a>**Q20.** Scan within a warp; collecting elements without gaps, with `__ballot_sync` and `__popc`. *([Day 8](day08/README.md))*

<a id="q21"></a>**Q21.** `__activemask` and `__syncwarp`: which lanes are actually present, and when an explicit warp barrier is needed. *([Day 9](day09/README.md))*

<a id="q22"></a>**Q22.** Atomic operations and `atomicCAS`; how they differ from a plain `*p += v`. *([Day 9](day09/README.md))*

<a id="q23"></a>**Q23.** Atomic contention, privatization, warp aggregation. *([Day 9](day09/README.md))*

<a id="q24"></a>**Q24.** Matrix multiplication: naive and tiled; global traffic, TFLOP/s. *([Day 10](day10/README.md))*

<a id="q25"></a>**Q25.** Hamming distance and descriptor matching; memory layout. *([Day 10](day10/README.md))*

<a id="q26"></a>**Q26.** Texture memory and the texture cache; texture objects. *([Day 11](day11/README.md))*

<a id="q27"></a>**Q27.** Texture filtering and addressing; the `0.5` offset; surfaces. *([Day 11](day11/README.md))*

<a id="q28"></a>**Q28.** CUDA graph: the DAG model, recording, instantiation, launch. *([Day 12](day12/README.md))*

<a id="q29"></a>**Q29.** Nodes and edges: which operations become nodes and which do not, and where the edges come from. *([Day 12](day12/README.md))*

<a id="q30"></a>**Q30.** Stream capture: fork and join with events, and the calls forbidden during capture. *([Day 12](day12/README.md))*

<a id="q31"></a>**Q31.** Launch overhead; when a graph pays off and when it does not. *([Day 12](day12/README.md))*

<a id="q32"></a>**Q32.** Matrix transpose through shared memory; the role of `[32][33]`. *([Day 12](day12/README.md))*

<a id="q33"></a>**Q33.** L1 and L2 caches, lines and sectors; the resource shared with shared memory. *([Day 13](day13/README.md))*

<a id="q34"></a>**Q34.** Cache hints, `__ldg`, L2 persistence. *([Day 13](day13/README.md))*

<a id="q35"></a>**Q35.** Register spilling and thread coarsening. *([Day 13](day13/README.md))*

<a id="q36"></a>**Q36.** Theoretical peak bandwidth, arithmetic intensity, the roofline. *([Day 13](day13/README.md))*

<a id="q37"></a>**Q37.** Correct measurement method: warm-up, averaging, CUDA events. *([Day 13](day13/README.md))*

<a id="q38"></a>**Q38.** Cooperative groups and dynamic parallelism. *([Day 14](day14/README.md))*

<a id="q39"></a>**Q39.** Stream-ordered allocation: guarantees within and across streams. *([Day 15](day15/README.md))*

<a id="q40"></a>**Q40.** Memory pools: release threshold, reserved and used memory; allocation inside a graph. *([Day 15](day15/README.md))*

---

## Practical part: defence of the independent assignment

During the semester every student is assigned one of the tasks below and solves it independently. The solution is emailed to the lecturer before the exam as source code and a short report: how the result was verified against the reference and how many elements differ, a table of `cudaEvent` times averaged over many runs, achieved throughput as a percentage of the GPU’s theoretical peak, and half a page explaining the performance. At the exam the program is run, verified and timed, and the student explains its performance and answers questions on the code.

Three of the tasks – connected components (E1), histogram (E2) and Canny (E3) – are specified in full, with minimum and extended requirements and typical mistakes, in [TASKS.md § Final Exam](TASKS.md#եզրափակիչ-քննություն).

1. Canny edge detector: Gaussian blur, Sobel gradient, non-maximum suppression, double threshold and hysteresis; comparison with `cv::Canny`.
2. Connected-components extraction on binary images: label propagation and union-find; comparison with `cv::connectedComponents`.
3. Separable filters (Gaussian, box): a horizontal and a vertical pass compared with a single tiled 2D pass.
4. Dilation and erosion with a configurable structuring element; opening and closing.
5. Matrix multiplication taking cache behaviour into account: tiling, coarsening, L2 reuse; percentage of peak and comparison with cuBLAS.
6. Mean and standard deviation of very large vectors, including vectors larger than device memory (streamed chunks, numerically stable merging of partial results).
7. Inclusive and exclusive scan (prefix sum) over a large vector and its use for removing elements that satisfy a predicate (stream compaction); comparison with a host version.
8. Integral image (summed-area table): row and column scans, box-filter sums in constant time per pixel; comparison with `cv::integral`.
9. DFT of a small fixed-size vector (e.g. 128 samples): direct computation and a radix-2 FFT with shared memory or warp shuffles; comparison with a host version.
10. Image pyramid construction: a multi-level Gaussian pyramid (`pyrDown`) and Laplacian pyramid (`pyrUp` and difference) with reconstruction of the original image; comparison with `cv::cuda::pyrDown`/`pyrUp`.
11. Histogram and histogram equalization (privatization, scan, look-up table).
12. Median filter with a sorting network in shared memory.
13. Bilateral (edge-preserving) filter.
14. Fixed and adaptive thresholding.
15. Perspective warp (homography) through textures.
16. Harris corner detection with non-maximum suppression.
17. Real-time motion detection on a camera or video stream (frame differencing, thresholding, morphological clean-up) with streams and a CUDA graph.

---

## Preparing

- The material for each question is in the **Definitions** and **Key Concepts** sections of the matching `dayNN/README.md`.
- Terms are in [GLOSSARY.md](GLOSSARY.md), function interfaces in [API.md](API.md), device intrinsics in [INTRINSICS.md](INTRINSICS.md).
- The performance questions (Q33–Q37) rely on [PERFORMANCE.md](PERFORMANCE.md).
- No CUDA machine? Use the [course Colab notebook](https://colab.research.google.com/drive/1zDtYkz8WwD7sOIucSyUoxm2n7RYWJxVZ?usp=sharing).
