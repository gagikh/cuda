# CUDA course (Compute Unified Device Architecture)
Title: Fundamentals of Accelerated Computation Using CUDA C/C++
**Armenian Slavonic University – Lectures & Labs (14 Days)**  
Instructor: Gagik Hakobyan

---

## Course Outline

### **Day 1: CUDA Basics and Programming Model**
- CUDA programming model overview  
- Host vs Device  
- GPU architecture fundamentals  
- Thread hierarchy overview

---

### **Day 2: Thread Hierarchy & Execution Model**
- SIMD architecture  
- Threads, blocks, grids: structure and enumeration  
- Launch configuration and kernel invocation  
- Thread indexing patterns

---

### **Day 3: Warp-Level Execution and Control Flow**
- Warp definition and behavior  
- Control flow: `if`, `else`, `for`, `while`  
- Loop unrolling (manual and compiler-assisted)  
- Divergence impact and avoidance

---

### **Day 4: CUDA Memory Types and Management**
- Paged, pinned (page-locked), and mapped memory  
- Unified memory (`cudaMallocManaged`)  
- Memory allocation strategies

---

### **Day 5: Memory Conflicts and Shared Memory**
- Bank conflicts in shared memory  
- Synchronized memory access inside kernels  
- Shared, constant, and pitched memory  
- Memory padding techniques

---

### **Day 6: Streams and Events**
- Global memory usage in real applications  
- Streams for concurrent execution  
- Events for timing and synchronization  
- Streamed read/write patterns

---

### **Day 7: Asynchronous Execution Techniques**
- `cudaMemcpy`: sync vs async  
- Asynchronous kernel launches  
- Stream priorities and dependencies  
- Event-based execution timing

---

### **Day 8: Warp-Level Intrinsics – Reduction**
- Warp shuffle functions (`__shfl_*`)  
- Intra-warp communication  
- Parallel reduction techniques  
- Performance tuning

---

### **Day 9: Warp-Level Data Exchange**
- Inter-thread data exchange  
- Warp vote functions (`__ballot_sync`, etc.)  
- Advanced warp shuffle patterns  
- Cooperative operations

---

### **Day 10: Practical Algorithms**
- Feature descriptor matching using Hamming distance  
- Bitwise operations in CUDA  
- Matrix multiplication (thread tiling, shared memory)

---

### **Day 11: Textures and Surfaces**
- Using textures for 2D spatial data  
- Surface memory usage  
- Texture filtering and addressing modes  
- Zoom and image processing example

---

### **Day 12: CUDA Graph API**
- Graph creation and capture  
- Recording kernels and memory operations  
- Launching graphs  
- Use cases for dynamic workloads

---

### **Day 13: Cache Behavior and Optimization**
- L1 and L2 cache control  
- Cache tuning APIs  
- Persistent L2 cache example  
- Measuring memory throughput

---

### **Day 14: CUDA Libraries and Applications**
- cuRAND: random number generation  
- cuBLAS: linear algebra basics  
- cuFFT: FFT usage  
- Estimating π using Monte Carlo with cuRAND

---

## Links


// Oxford course link
https://people.maths.ox.ac.uk/~gilesm/cuda/

// labs
https://iis-people.ee.ethz.ch/~gmichi/asocd_2014/exercises/ex_03.pdf
// lectures from the link

// Nvidia official c programming guide
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf

// Programming model
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model

// Parallel programming with CUDA
http://www.davidmuench.de/studienarbeit.pdf

// 

// Intorduction
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_01.pdf
- tasks, trivial vector addition example

// Different memory and variable types - basic kernel implementation
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_02.pdf

// control flow, atomics
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_03.pdf

// warp based programming model -> too complex after lecture 2, need to do computational tasks
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_04.pdf

// libraries, reomve this topic and provide computational tasks
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_05.pdf

// streams and host related code
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_06.pdf

// Usefull links
http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf

// Nvidia lectures
https://developer.nvidia.com/educators/existing-courses#2


// cuda-gdb
set cuda break_on_launch application
cuda device sm warp lane block thread
//step

## 📝 CUDA Exam Topics

The final exam covers both theory and practice based on the following core CUDA programming topics:

- **Kernels and Launch**
  - Kernel declaration and invocation
  - Thread indexing and grid configuration

- **Warp and Operations**
  - Warp execution model
  - Control flow and warp divergence
  - Warp shuffle and vote functions

- **Shared Memory**
  - Shared memory declaration and access
  - Synchronization with `__syncthreads()`
  - Optimization techniques and bank conflict avoidance

- **Paged and Pinned Memory**
  - Differences between paged and pinned memory
  - Performance implications and allocation methods

- **Atomic Operations and Global Memory**
  - Atomic functions in global memory
  - Use cases for avoiding race conditions

- **Mapped Memory**
  - Zero-copy memory
  - Host-device memory mapping using `cudaHostAlloc`

- **Memory Transfers and Sync/Async Launch**
  - `cudaMemcpy` synchronous and asynchronous usage
  - Stream-based memory copy and overlap with execution

- **Streams and Events, Synchronization**
  - Creating and managing CUDA streams
  - Events for timing and synchronization
  - Stream dependencies

- **CUDA Graphs**
  - Graph creation, capture, and launch
  - Performance benefits and use cases

- **Texture Memory and Binding**
  - Texture reference and binding to arrays
  - Filtering, addressing modes
  - Use in image processing and interpolation

- **Bank Conflicts and Cache Control**
  - Shared memory bank conflict causes and solutions
  - L1/L2 cache tuning and control APIs

---

> **Note:** All exam questions are based on materials covered in lectures and labs. Hands-on familiarity with coding patterns, memory strategies, and debugging tools is essential.
