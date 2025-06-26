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

Exam notes
- kernels and launch
- warp and operations
- shared memory
- paged/pinned memory
- atomic operations and global memory
- mapped memory
- memory transfers, sync/async launch
- streams and events, synchronization
- graph and graph record
- texture memory and binding
- bank conflicts and cache control

Tasks:
- opencv filters and optimizations - remove kernel matrices and provide parameters instead
- Compute Beblid descriptors with cuda - https://github.com/opencv/opencv_contrib/blob/80f1ca2442982ed518076cd88cf08c71155b30f6/modules/xfeatures2d/src/beblid.cpp
- optimze RANSAC taking into account that camera moves toward to the scene (use cuda to solve liear equations)
- given 
1) reference objects T1, T2, .. Tn, 
2) some operator <, that T1 < T2 < ... Tn
3) L1 < L2 < ..... Lm
4) P = {Pij, i = 1, ...m, j = 1, ... m} and Pij is the probability that Li is close to Tj
5) find assignement vector V, such that sum(P(V)) -> min (solve with cuda) 
// comment: this is kuhn muknres problem with restrictions, if L[i] is assigned to T[j], then L[i + 1] can be assigned to T[j+1], ... T[n] only
// sould solved for large Ps, so cuda is needed.
- Transformer TrTr with batches ??
