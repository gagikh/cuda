# CUDA Course (Compute Unified Device Architecture)

**Title**: Fundamentals of Accelerated Computation Using CUDA C/C++  
**University**: Armenian Slavonic University – Lectures & Labs (15 Days)  
**Instructor**: Gagik Hakobyan

---

## 📘 Course Outline

### [Day 1: CUDA Basics and Programming Model](day1/README.md)
- CUDA programming model overview  
- Host vs Device  
- GPU architecture fundamentals  
- Thread hierarchy overview

### [Day 2: Thread Hierarchy & Execution Model](day2/README.md)
- SIMD architecture and instructions pipeline
- Threads, blocks, grids: structure and enumeration  
- Launch configuration and kernel invocation  
- Thread indexing patterns

### [Day 3: Warp-Level Execution and Control Flow](day3/README.md)
- Warp definition and behavior  
- Control flow: `if`, `else`, `for`, `while`  
- Loop unrolling  
- Divergence impact and avoidance

### [Day 4: CUDA Memory Types and Management](day4/README.md)
- Paged, pinned, and mapped memory  
- Unified memory  
- Allocation strategies

### [Day 5: Memory Conflicts and Shared Memory](day5/README.md)
- Bank conflicts  
- Synchronized memory access  
- Shared, constant, and pitched memory  
- Memory padding

### [Day 6: Streams and Events](day6/README.md)
- Global memory usage  
- Streams and concurrent execution  
- Events and synchronization  
- Streamed read/write patterns

### [Day 7: Asynchronous Execution Techniques](day7/README.md)
- `cudaMemcpy`: sync vs async  
- Async kernel launches  
- Stream dependencies  
- Event-based timing

### [Day 8: Warp-Level Intrinsics – Reduction](day8/README.md)
- Warp shuffle functions  
- Intra-warp communication  
- Parallel reduction  
- Performance tuning

### [Day 9: Warp-Level Data Exchange](day9/README.md)
- Warp vote functions  
- Inter-thread data exchange  
- Cooperative operations

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

### [Day 14: CUDA Libraries](day14/README.md)
- cuRAND (random generation)  
- cuBLAS (linear algebra)  
- cuFFT (FFT)  
- Monte Carlo π estimation

### [Day 15: Stream-Ordered Memory Allocation](day15/README.md)
- `cudaMallocAsync` / `cudaFreeAsync`  
- Stream-ordered allocation semantics  
- Memory pools

---

## 📚 Recommended Resources

### CUDA Programming Guides
- [CUDA C Programming Guide (HTML)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
- [CUDA C Programming Guide (PDF)](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)

### Oxford CUDA Course by Mike Giles
- [Main Page](https://people.maths.ox.ac.uk/~gilesm/cuda/)
- [Lecture 1 – Introduction & Vector Add](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_01.pdf)  
- [Lecture 2 – Memory & Kernel Basics](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_02.pdf)  
- [Lecture 3 – Control Flow & Atomics](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_03.pdf)  
- [Lecture 4 – Warp Programming (Advanced)](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_04.pdf)  
- [Lecture 5 – Libraries (Skip)](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_05.pdf)  
- [Lecture 6 – Streams & Host Code](https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_06.pdf)

### Labs & Exercises
- [ETH Zurich CUDA Labs (PDF)](https://iis-people.ee.ethz.ch/~gmichi/asocd_2014/exercises/ex_03.pdf)

### Extra Reading
- [CUDA by Example](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)  
- [Parallel Programming with CUDA (David Muench)](http://www.davidmuench.de/studienarbeit.pdf)

### NVIDIA Course Materials
- [NVIDIA Educator Courses](https://developer.nvidia.com/educators/existing-courses#2)

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
