# CUDA Course (Compute Unified Device Architecture)

**Title**: Fundamentals of Accelerated Computation Using CUDA C/C++  
**University**: Armenian Slavonic University – Lectures & Labs (15 Days)  
**Instructor**: Gagik Hakobyan

📖 [Glossary](GLOSSARY.md) — the terms this course uses, alphabetical, tagged with the day it's introduced.  
⚡ [Intrinsics Cheat Sheet](INTRINSICS.md) — warp shuffle/vote, bit ops, math, cache hints, atomics and barriers in one table, tagged by day.  
📋 [100 Practice Tasks](TASKS.md) — every day's Self-Learning tasks in one list, plus 25 bonus tasks beyond the 15-day structure.  
🖥️ [Architecture Deep Dive](ARCHITECTURE.md) — what's actually inside an SM: registers, ALUs, FPUs, tensor cores, and how shared/constant/L2/global memory are organized.  
🚀 [Performance Checklist](PERFORMANCE.md) — coalescing, occupancy, roofline, privatization, coarsening: which optimization to try, in what order, and how to know when to stop.  
🎬 [SM Animations](sm_animations.html) — interactive companion to the architecture doc (open locally; GitHub shows HTML as source).

---

## 💻 No CUDA GPU? Start here

**[▶ Open the course Colab notebook](https://colab.research.google.com/drive/1zDtYkz8WwD7sOIucSyUoxm2n7RYWJxVZ?usp=sharing)**

You do not need a CUDA-capable machine to take this course. The notebook above gives you a free NVIDIA GPU in the browser and is the baseline environment for every lab — clone the repo into it, paste a day's `template.cu` into a cell, compile and run.

Set it up once: **Runtime → Change runtime type → Hardware accelerator → GPU**, then confirm you actually have one:

```
!nvcc --version
!nvidia-smi
```

Three things to know before you rely on it:

- **The GPU is usually a Tesla T4**, compute capability 7.5 — which is exactly the `-arch=sm_75` this course compiles for, so every template builds unmodified. Run Day 1's `report_device_capabilities()` first and write down *your* session's numbers; a free session can also hand you a different card.
- **Days 5 onward need OpenCV built with CUDA.** Colab's preinstalled `opencv-python` is CPU-only, so `cv::cuda::GpuMat` will not link. Either build OpenCV with `-DWITH_CUDA=ON` in the notebook (slow, but done once per session), or replace the image I/O with a plain `cudaMalloc` buffer and a synthetic image — the CUDA content of each day is unaffected either way.
- **Sessions are ephemeral.** Anything not saved to Drive disappears when the runtime recycles, and idle sessions are reclaimed. Keep your work in a Drive-mounted folder or a GitHub fork.

Local builds are still preferred where you have the hardware — profiling with Nsight Systems and Nsight Compute (Days 4, 13 and [PERFORMANCE.md](PERFORMANCE.md)) is far more usable on a real desktop.

---

## 📘 Course Outline

### [Day 1: CUDA Basics and Programming Model](day01/README.md)
- CUDA programming model overview  
- Host vs Device  
- GPU architecture fundamentals  
- Thread hierarchy overview

### [Day 2: Thread Hierarchy & Execution Model](day02/README.md)
- Threads, blocks, grids: structure and enumeration  
- Launch configuration and kernel invocation  
- Thread indexing patterns  
- Memory coalescing  
- Occupancy and block-size choice  
- Grid-stride loops

### [Day 3: Warp-Level Execution and Control Flow](day03/README.md)
- SIMD architecture and the instruction pipeline  
- Warp definition and behavior  
- Control flow: `if`, `else`, `for`, `while`  
- Loop unrolling  
- Divergence impact and avoidance

### [Day 4: CUDA Memory Types and Management](day04/README.md)
- Paged, pinned, and mapped memory  
- Unified memory  
- Allocation strategies

### [Day 5: Memory Conflicts and Shared Memory](day05/README.md)
- Bank conflicts  
- Synchronized memory access  
- Shared, constant, and pitched memory  
- Memory padding

### [Day 6: Streams and Events](day06/README.md)
- Global memory usage  
- Streams and concurrent execution  
- Events and synchronization  
- Streamed read/write patterns

### [Day 7: Asynchronous Execution Techniques](day07/README.md)
- `cudaMemcpy`: sync vs async  
- Async kernel launches  
- Stream dependencies  
- Event-based timing

### [Day 8: Warp-Level Intrinsics – Reduction](day08/README.md)
- Warp shuffle functions; the lane mask and why divergence here is UB, not just slow  
- Intra-warp communication without shared memory or barriers  
- Parallel reduction: warp-level, then block-level (warp → shared → warp)  
- Inclusive scan, stream compaction, and warp-aggregated atomics  
- Performance tuning: shuffles vs. shared memory

### [Day 9: Warp-Level Data Exchange](day09/README.md)
- Warp vote functions  
- Inter-thread data exchange  
- Cooperative operations  
- Atomic contention and privatization

### [Day 10: Practical Algorithms](day10/README.md)
- Hamming distance matching  
- Bitwise ops  
- Matrix multiplication

### [Day 11: Textures and Surfaces](day11/README.md)
- Texture memory  
- Surface memory  
- Filtering & addressing  
- Zoom/image processing

### [Day 12: CUDA Graph API](day12/README.md)
- Graph recording  
- Kernel + memory op capture  
- Graph launch

### [Day 13: Cache Behavior and Optimization](day13/README.md)
- L1/L2 cache  
- Persistent cache  
- Memory throughput  
- Thread coarsening  
- Memory-bound vs. compute-bound; % of peak bandwidth

### [Day 14: CUDA Libraries](day14/README.md)
- cuRAND (random generation)  
- cuBLAS (linear algebra)  
- cuFFT (FFT)  
- Monte Carlo π estimation

### [Day 15: Stream-Ordered Memory Allocation](day15/README.md)
- `cudaMallocAsync` / `cudaFreeAsync`  
- Stream-ordered allocation semantics; cross-stream use needs an explicit event  
- Memory pools: default vs. explicit, release threshold, reserved/used, trimming  
- Reuse policies, IPC pools, and why `cudaMalloc` can't be graph-captured

---

## 📚 Bibliography

### Primary

- NVIDIA. *CUDA C Programming Guide* — [PDF](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) · [HTML](https://docs.nvidia.com/cuda/cuda-programming-guide/)  
  The course's primary reading. Each day's `Resources` section names the chapters for that session.
- NVIDIA. *CUDA C++ Best Practices Guide* — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Hwu W., Kirk D., El Hajj I. *Programming Massively Parallel Processors*, 5th ed., Elsevier, 2026  
  The standard text. Chapters 5–6 cover the same ground as [PERFORMANCE.md](PERFORMANCE.md).

### Profiling and performance

- NVIDIA. *Nsight Compute* — https://docs.nvidia.com/nsight-compute/
- NVIDIA. *Nsight Systems* — https://docs.nvidia.com/nsight-systems/
- Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model. *CACM* 52(4), 2009

### Libraries and precision

- cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND — https://docs.nvidia.com/cuda/
- cuDNN — https://docs.nvidia.com/deeplearning/cudnn/
- CUB — https://nvidia.github.io/cccl/cub/ · Thrust — https://nvidia.github.io/cccl/thrust/
- NVIDIA. *Train With Mixed Precision* — https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740

### Computer vision

Days 5 onward operate on real images through OpenCV.

- OpenCV CUDA module — https://docs.opencv.org/4.x/d1/d1e/group__cuda.html
- Szeliski R. *Computer Vision: Algorithms and Applications*, 2nd ed. Free PDF: https://szeliski.org/Book/

### Courses and teaching material

- NVIDIA DLI Teaching Kit — Accelerated Computing. https://developer.nvidia.com/teaching-kits
- NVIDIA / OLCF CUDA Training Series — https://www.olcf.ornl.gov/cuda-training-series/
- Oxford CUDA course, Mike Giles — https://people.maths.ox.ac.uk/~gilesm/cuda/
- GPU MODE lecture series — https://github.com/gpu-mode/lectures

---

## 🛠 CUDA Debugging Tips

```bash
# Enable debugging and break on kernel launch
cuda-gdb
set cuda break_on_launch application
cuda device sm warp lane block thread
# Use 'step' to go line by line
```

---

## 📝 CUDA Exam Topics

The final exam covers both theory and practical knowledge. Key areas include:

- **Kernels & Launch** — syntax, launch parameters, thread indexing  
- **Coalescing & Occupancy** — warp-level access patterns, block-size choice, latency hiding  
- **Performance Method** — memory-bound vs compute-bound, achieved vs. theoretical bandwidth  
- **Warp & Operations** — warp execution, divergence, shuffle/vote intrinsics  
- **Shared Memory** — access, `__syncthreads()`, optimization  
- **Paged vs Pinned Memory** — allocation, performance  
- **Atomic Ops & Global Memory** — preventing race conditions  
- **Mapped Memory** — zero-copy, host/device mapping  
- **Memory Transfers & Async Execution** — `cudaMemcpy`, stream overlap  
- **Streams & Events** — concurrency, timing, dependencies  
- **CUDA Graphs** — record, launch, optimize workflows  
- **Texture Memory** — filtering, binding, addressing  
- **Bank Conflicts & Cache** — tuning L1/L2, avoiding conflicts

🧠 Tip: Practice writing and debugging CUDA kernels. Focus on memory strategies and performance tuning.
