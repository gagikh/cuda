# Glossary

Terms used across this course, alphabetically. Each entry notes the day it's first introduced (`(Day N)`) — that's where to look for the full explanation and a diagram, if there is one.

**Bank conflict** — When multiple threads in a warp access shared memory addresses that map to the same bank in the same transaction, forcing those accesses to be serialized instead of serviced in parallel. Fixed by padding (Day 5) or index swizzling (Day 13). *(Day 5)*

**Block** — A group of threads (up to `maxThreadsPerBlock`, see `report_device_capabilities()`) that execute on the same SM and can cooperate via shared memory and `__syncthreads()`. A kernel launch creates a grid of blocks. *(Day 2)*

**Compute capability** — A version number (e.g. `8.6`) identifying a GPU's architecture generation and feature set. Written as `sm_XX`/`compute_XX` in nvcc flags — see **PTX** and **SASS** below. *(Day 1)*

**CUDA_CHECK / CUDA_CHECK_LAST_ERROR** — This course's macros (`common/cuda_check.h`) for checking a CUDA API call's return value, and for checking whether the most recent kernel launch failed (launches themselves return nothing). *(Day 1)*

**Device** — The GPU, as opposed to the **host** (CPU). Has its own memory space (VRAM), reached over PCIe/NVLink. *(Day 1)*

**Divergence (warp divergence)** — When threads within the same warp take different paths through a branch (`if`/`else`, varying loop counts). Because a warp executes in lockstep, the hardware runs each path separately with some lanes masked off, instead of truly in parallel — a direct performance cost. *(Day 3)*

**Grid** — The full set of blocks launched by one kernel call (`<<<grid, block>>>`). *(Day 2)*

**Grid-stride loop** — A launch pattern where a fixed number of threads each process multiple elements in a loop, striding by the total thread count, instead of sizing the grid to exactly match the data. Handles any input size without recomputing launch dimensions. *(Day 2)*

**GpuMat (`cv::cuda::GpuMat`)** — OpenCV's device-side image/matrix type, the GPU counterpart of `cv::Mat`. Rows are pitched (see **pitch**), not necessarily contiguous. *(Day 5)*

**Hamming distance** — The number of differing bits between two equal-length binary values, computed efficiently on GPU via `a ^ b` followed by `__popc` (population count). Used for matching binary feature descriptors (e.g. ORB). *(Day 10)*

**Host** — The CPU, as opposed to the **device** (GPU). *(Day 1)*

**Kernel** — A function marked `__global__`, launched from host code with `<<<grid, block>>>` syntax, executed by many threads in parallel on the device. *(Day 1)*

**Latency hiding** — The GPU's core performance strategy: when one warp stalls (e.g. waiting on a slow memory load), the warp scheduler issues an instruction from a *different*, ready warp that same cycle instead of leaving the pipeline idle. Why GPUs favor many threads over few fast ones. *(Day 3)*

**Occupancy** — Roughly, how many warps are resident on an SM at once relative to the maximum it could hold. Limited by whichever resource runs out first per block: registers, shared memory, or thread-count limits (all visible via `report_device_capabilities()`). *(Day 1, revisited Day 6+)*

**Pinned memory** — Host memory that's been page-locked (`cudaMallocHost`/`cudaHostRegister`), so the OS can't move it and the GPU can DMA it directly — faster transfers, and required for true async `cudaMemcpyAsync` overlap. OpenCV's equivalent type is `cv::cuda::HostMem`. *(Day 4)*

**Pitch** — The actual byte stride between rows of a 2D allocation (`cudaMallocPitch`, or a `GpuMat`'s `.step`), which is normally larger than `width * elementSize` due to alignment padding. Kernels touching pitched memory must index rows by pitch, not by width. *(Day 5)*

**PTX** — NVIDIA's virtual, forward-compatible GPU assembly language. nvcc compiles device code to PTX first; `ptxas` then assembles PTX into real machine code (**SASS**) for a specific architecture. *(Day 1)*

**SASS** — The real machine code (cubin) for one specific GPU architecture, assembled from PTX by `ptxas`. *(Day 1)*

**SIMT (Single Instruction, Multiple Threads)** — NVIDIA's execution model: one instruction is fetched/decoded once and issued to all 32 threads of a warp simultaneously. See **divergence** for what happens when threads disagree on control flow. *(Day 3)*

**SM (Streaming Multiprocessor)** — A GPU's core compute unit; a modern GPU has dozens to over a hundred. Each block runs entirely on one SM. Real counts/limits for your GPU are in `report_device_capabilities()`. *(Day 1)*

**Stream** — An ordered queue of GPU operations (kernels, copies). Operations in different streams can run concurrently; operations within one stream execute in issue order. *(Day 6)*

**Stream-ordered allocation** — `cudaMallocAsync`/`cudaFreeAsync`, which tie allocation/deallocation to a stream instead of forcing a device-wide synchronization like classic `cudaMalloc`/`cudaFree` do. *(Day 15)*

**Swizzling** — Scrambling a shared-memory index (e.g. `tile[row][col ^ row]`) so that a fixed logical column maps to a different physical bank on every row, removing bank conflicts without wasting a padding column. *(Day 13)*

**Tensor Core** — Specialized SM hardware (Volta and newer, compute capability ≥ 7.0) for fast mixed-precision matrix-multiply-accumulate, used by libraries like cuBLAS. *(Day 1, Day 14)*

**Texture object** — A GPU resource bound to memory (often pitched 2D) that supports hardware-accelerated filtering (e.g. bilinear) and address clamping on read, accessed in-kernel via `tex2D`. *(Day 11)*

**Thread** — The smallest unit of execution; identified within its block by `threadIdx`, within the grid by combining `threadIdx` with `blockIdx`/`blockDim`. *(Day 1-2)*

**Warp** — A group of 32 threads within a block that the hardware schedules and executes together in lockstep (see **SIMT**). The unit that warp-level intrinsics (`__shfl_*`, `__ballot_sync`, ...) operate on. *(Day 3, Day 8-9)*
