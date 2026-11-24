# Final Exam · Day 1 – Day 15

The final exam covers the whole course, Day 1 through Day 15. The midterm ([MIDTERM.md](MIDTERM.md)) covers Day 8 – Day 15 only.

A ticket has **two theory questions and one practical task**. Both lists are published below in full, so students know in advance everything they can be asked.

**Marking (100 points).**

| Part | Points | What is assessed |
|---|---|---|
| Theory question 1 | 30 | Accurate definitions (15), explanation in your own words with an example or a calculation (10), answers to follow-up questions (5) |
| Theory question 2 | 30 | Same |
| Practical task (E1–E3) | 40 | Correctness against the OpenCV reference (20), error handling and a clean `compute-sanitizer` run (7), `cudaEvent` timing averaged over many runs and expressed as % of peak (8), explanation of the performance (5) |

A correct kernel that has not been measured does not get full marks, and neither does a fast one you cannot explain.

---

## Theory questions

<a id="q1"></a>**Q1.** The CUDA programming model: host and device, kernel, thread, block, grid. *([Day 1](day01/README.md))*

<a id="q2"></a>**Q2.** The thread hierarchy, built-in variables, computing the global index in 1D and 2D. *([Day 2](day02/README.md))*

<a id="q3"></a>**Q3.** Coalesced global-memory access; the number of transactions for sequential and strided access. *([Day 2](day02/README.md))*

<a id="q4"></a>**Q4.** Occupancy and the choice of block size; grid-stride loops. *([Day 2](day02/README.md))*

<a id="q5"></a>**Q5.** The SIMT model, the warp, the instruction pipeline, latency hiding. *([Day 3](day03/README.md))*

<a id="q6"></a>**Q6.** Branch divergence and reconvergence; loop unrolling. *([Day 3](day03/README.md))*

<a id="q7"></a>**Q7.** Pageable, pinned, mapped and unified host memory. *([Day 4](day04/README.md))*

<a id="q8"></a>**Q8.** Shared memory: banks, conflicts, broadcast, __syncthreads(). *([Day 5](day05/README.md))*

<a id="q9"></a>**Q9.** Removing bank conflicts: padding and XOR swizzling. *([Day 5](day05/README.md))*

<a id="q10"></a>**Q10.** Constant and pitched memory. *([Day 5](day05/README.md))*

<a id="q11"></a>**Q11.** CUDA streams and events; device-side timing. *([Day 6](day06/README.md))*

<a id="q12"></a>**Q12.** Asynchronous copies, double buffering, inter-stream dependencies. *([Day 7](day07/README.md))*

<a id="q13"></a>**Q13.** Warp and lane; the lane mask; shuffle functions and the width parameter. *([Day 8](day08/README.md))*

<a id="q14"></a>**Q14.** Warp- and block-level reduction; the warp → shared → warp scheme. *([Day 8](day08/README.md))*

<a id="q15"></a>**Q15.** Scan within a warp; compaction with __ballot_sync and __popc. *([Day 8](day08/README.md))*

<a id="q16"></a>**Q16.** Atomic operations, atomicCAS; how they differ from a plain *p += v. *([Day 9](day09/README.md))*

<a id="q17"></a>**Q17.** Atomic contention, privatization, warp aggregation. *([Day 9](day09/README.md))*

<a id="q18"></a>**Q18.** Matrix multiplication: naive and tiled; global traffic, TFLOP/s. *([Day 10](day10/README.md))*

<a id="q19"></a>**Q19.** Hamming distance and descriptor matching; memory layout. *([Day 10](day10/README.md))*

<a id="q20"></a>**Q20.** Texture memory and the texture cache; texture objects. *([Day 11](day11/README.md))*

<a id="q21"></a>**Q21.** Texture filtering and addressing; the 0.5 offset; surfaces. *([Day 11](day11/README.md))*

<a id="q22"></a>**Q22.** CUDA graph: the DAG model, recording, instantiation, launch. *([Day 12](day12/README.md))*

<a id="q23"></a>**Q23.** Stream capture: fork and join with events, forbidden calls. *([Day 12](day12/README.md))*

<a id="q24"></a>**Q24.** Launch overhead; when a graph pays off. *([Day 12](day12/README.md))*

<a id="q25"></a>**Q25.** Matrix transpose through shared memory; the role of [32][33]. *([Day 12](day12/README.md))*

<a id="q26"></a>**Q26.** L1 and L2 caches, lines and sectors; the resource shared with shared memory. *([Day 13](day13/README.md))*

<a id="q27"></a>**Q27.** Cache hints, __ldg, L2 persistence. *([Day 13](day13/README.md))*

<a id="q28"></a>**Q28.** Register spilling and thread coarsening. *([Day 13](day13/README.md))*

<a id="q29"></a>**Q29.** Theoretical peak bandwidth, arithmetic intensity, the roofline. *([Day 13](day13/README.md))*

<a id="q30"></a>**Q30.** Correct measurement method: warm-up, averaging, CUDA events. *([Day 13](day13/README.md))*

<a id="q31"></a>**Q31.** FMA, fp64/fp32/tf32/bf16/fp16, mixed precision, tensor cores. *([Day 14](day14/README.md))*

<a id="q32"></a>**Q32.** Cooperative groups and dynamic parallelism. *([Day 14](day14/README.md))*

<a id="q33"></a>**Q33.** Stream-ordered allocation: guarantees within and across streams. *([Day 15](day15/README.md))*

<a id="q34"></a>**Q34.** Memory pools: release threshold, reserved and used memory; allocation inside a graph. *([Day 15](day15/README.md))*

---

## Practical tasks

The practical part is one of three computer-vision algorithms, specified in full in [TASKS.md § Final Exam](TASKS.md#եզրափակիչ-քննություն). The variant is assigned in advance; the student solves it independently and defends it at the exam: running it, verifying it against the OpenCV reference, presenting the timing report and explaining the performance orally.

| | Task | What it tests |
|---|---|---|
| **E1** | Connected components labeling of a binary mask | Iterative kernels, atomics, halo handling, convergence on the device |
| **E2** | Histogram of a grayscale image | Atomic contention and privatization, warp aggregation |
| **E3** | Canny edge detection | A multi-stage pipeline: tiling, per-stage timing, fusion, streams |

---

## Preparing

- The material for each question is in the **Definitions** and **Key Concepts** sections of the matching `dayNN/README.md`.
- Terms are in [GLOSSARY.md](GLOSSARY.md), function interfaces in [API.md](API.md), device intrinsics in [INTRINSICS.md](INTRINSICS.md).
- The performance questions (Q26–Q30) rely on [PERFORMANCE.md](PERFORMANCE.md).
- No CUDA machine? Use the [course Colab notebook](https://colab.research.google.com/drive/1zDtYkz8WwD7sOIucSyUoxm2n7RYWJxVZ?usp=sharing).
