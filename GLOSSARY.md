# Glossary

Terms used across this course, alphabetically. Each entry notes the day it is introduced, `(Day N)` — that's where to look for the full explanation and a diagram, if there is one. For the device-side *functions* rather than the concepts, see [INTRINSICS.md](INTRINSICS.md); for the hardware behind them, [ARCHITECTURE.md](ARCHITECTURE.md); for when each technique applies, [PERFORMANCE.md](PERFORMANCE.md).

**Achieved bandwidth** — Bytes a kernel actually moves divided by its runtime, normally expressed as a percentage of theoretical peak. Computed by `kernel_timer_t::gb_per_s()` in [`common/timer.h`](common/timer.h). *(Day 13)*

**`__activemask`** — Returns which lanes are converged at this instruction. It reports what happens to be true, and is not a substitute for a mask the code determines itself. *(Day 9)*

**Arithmetic intensity** — FLOPs performed per byte of memory traffic. Decides which side of the roofline a kernel sits on: low intensity means memory-bound, which covers most kernels; high means compute-bound. Tiling raises it without changing the arithmetic. *(Day 13)*

**Atomic contention** — Several threads targeting the same address. The updates serialize at L2, so the cost grows with the number of colliding threads, not with the number of atomic instructions. *(Day 9)*

**Atomic operation** — A read-modify-write on one address that no other thread can interleave with: `atomicAdd`, `atomicCAS`, `atomicMax` and the rest. *(Day 9)*

**`__ballot_sync`** — Returns a 32-bit mask with bit N set if lane N's predicate was true, delivered to every participating lane. *(Day 9)*

**Bank** — One of the 32 equal divisions of shared memory. Successive 32-bit words fall in successive banks, and each bank serves one word per cycle. *(Day 5)*

**Bank conflict** — When several threads of a warp access shared memory addresses mapping to the same bank in one transaction, those accesses are serialized instead of served in parallel. Fixed by padding (Day 5) or by index swizzling (Day 13). *(Day 5)*

**Barrier** — A point all participating threads must reach before any may continue. `__syncthreads()` is the block-wide barrier, `__syncwarp` the warp-wide one. *(Day 5)*

**Block** — A group of threads, up to `maxThreadsPerBlock`, that execute on the same SM and can cooperate through shared memory and `__syncthreads()`. A kernel launch creates a grid of blocks. *(Day 2)*

**`blockIdx`, `threadIdx`, `blockDim`, `gridDim`** — Built-in read-only variables in device code: the block's index in the grid, the thread's index in the block, the block's dimensions, the grid's dimensions. *(Day 2)*

**Broadcast** — When every lane of a warp reads the same shared memory word, the hardware serves them in one transaction. This is not a conflict. *(Day 5)*

**Cache line** — The 128-byte unit of L1 allocation, and the reason a coalesced warp access costs one transaction. *(Day 2, revisited Day 13)*

**Cache operator** — A per-instruction hint about how a load or store should use the caches: `__ldg` for the read-only path, `__ldcs` for a streaming load that is evicted first, `__stcs` for a streaming store, `__ldlu` for a last-use load whose line is then discarded. *(Day 13)*

**Coalescing (memory coalescing)** — When the 32 lanes of a warp access consecutive addresses, the hardware serves them in a single 128-byte transaction instead of up to 32 separate ones. The property belongs to the *warp*, not the thread: what matters is the combined footprint of one instruction across all 32 lanes, not the pattern one thread traces over time. *(Day 2, measured Day 13)*

**Compute-bound and memory-bound** — Whether a kernel's ceiling is instruction throughput or memory bandwidth. Determines which optimizations can possibly help; see **roofline**. *(Day 13)*

**Compute capability** — A version number, for example `8.6`, identifying a GPU's architecture generation and feature set. Written as `sm_XX` and `compute_XX` in nvcc flags. *(Day 1)*

**Constant memory** — A 64 KB read-only region declared `__constant__`, cached per SM. When all lanes of a warp read the same address it is served as one broadcast; divergent addresses are serialized. *(Day 5)*

**Cooperative groups** — An API that makes the group a piece of code synchronizes over explicit — the block, the currently converged lanes, the whole grid — instead of implicit in a hand-written lane mask. *(Day 14)*

**cuBLAS, cuFFT, cuRAND, cuDNN, NPP, nvJPEG, CUB, Thrust** — NVIDIA's libraries: dense linear algebra; fast Fourier transforms; random number generation; deep learning primitives; image and signal processing; JPEG decode and encode; block- and device-level parallel primitives; and an STL-like algorithms layer built on CUB. *(Day 14)*

**CUDA_CHECK / CUDA_CHECK_LAST_ERROR** — This course's macros ([`common/cuda_check.h`](common/cuda_check.h)) for checking a CUDA API call's return value, and for checking whether the most recent kernel launch failed — launches themselves return nothing. *(Day 1)*

**`cudaGetDeviceProperties`** — The API call returning a `cudaDeviceProp` structure with the device's SM count, warp size, per-SM register and shared memory limits, clock rates, memory bus width and compute capability. `report_device_capabilities()` in [`common/device_info.h`](common/device_info.h) prints it. *(Day 1)*

**CUDA graph** — A recorded directed acyclic graph of operations — kernels, copies, host callbacks — together with their dependencies, launched as one unit. *(Day 12)*

**`cudaHostRegister`** — Page-locks memory that was allocated normally, giving it the transfer properties of pinned memory without reallocating it. `cudaHostUnregister` reverses this. *(Day 4)*

**`cudaMemcpyAsync`** — A copy issued into a stream, returning immediately. It is genuinely asynchronous only when the host memory is page-locked; with pageable memory the runtime falls back to a synchronous copy and does not report it. *(Day 7)*

**`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — The runtime call returning how many blocks of a given kernel and block size will be resident per SM, without running the kernel. *(Day 2)*

**Default stream (per-thread)** — The stream used when none is named. Compiled with `--default-stream per-thread`, each host thread gets its own default stream, which does not serialize against other streams. *(Day 6)*

**Device** — The GPU, as opposed to the **host** (CPU). Has its own memory space (VRAM), reached over PCIe or NVLink. *(Day 1)*

**Device-side timing** — Measuring with events rather than a host clock, so the interval measured is the GPU's and excludes host-side launch latency. *(Day 6)*

**Divergence (warp divergence)** — When threads within one warp take different paths through a branch. Because a warp executes in lockstep, the hardware runs each path separately with some lanes masked off, instead of in parallel — a direct performance cost, and inside a `_sync` intrinsic a correctness one. *(Day 3)*

**DMA (Direct Memory Access)** — A transfer carried out by a copy engine without the CPU moving the data. It requires the host pages to be page-locked, which is why pinned transfers are faster. *(Day 4)*

**Double buffering** — Splitting the input into chunks and using two or more sets of buffers and streams, so that while chunk n is computed, chunk n+1 is copied in and chunk n−1 copied out. The ceiling becomes the larger of the transfer and compute rates rather than their sum. *(Day 7)*

**Eligible, active and stalled warp** — An active, that is resident, warp occupies a warp slot on the SM. It is eligible when its next instruction's operands and the required unit are ready, and stalled otherwise. The scheduler issues only from eligible warps. *(Day 3)*

**Event** — A marker placed in a stream with `cudaEventRecord`, which completes when all work preceding it in that stream completes. Used to wait (`cudaEventSynchronize`) and to measure (`cudaEventElapsedTime`). *(Day 6)*

**Fat binary** — The single executable nvcc produces, holding host machine code together with one or more device images (PTX, SASS, or both). At launch the driver picks a matching SASS image, or JIT-compiles the embedded PTX if none matches. *(Day 1)*

**FMA (fused multiply-add)** — `a * b + c` computed with a single rounding instead of two. One reason a GPU result and a CPU result can differ in the last bits for identical inputs in identical order. *(Day 14)*

**fp64, fp32, tf32, bf16, fp16** — Floating-point formats, given as exponent and mantissa bits. fp64: 11 and 52. fp32: 8 and 23. tf32: 8 and 10, a tensor core input format only. bf16: 8 and 7, the same range as fp32 with less precision. fp16: 5 and 10, narrower range and precision. *(Day 14)*

**GpuMat (`cv::cuda::GpuMat`)** — OpenCV's device-side image/matrix type, the GPU counterpart of `cv::Mat`. Rows are pitched (see **pitch**), not necessarily contiguous. *(Day 5)*

**Graph capture** — Recording a sequence of stream operations into a graph instead of executing it, between `cudaStreamBeginCapture` and `cudaStreamEndCapture`. *(Day 12)*

**Grid** — The full set of blocks launched by one kernel call, `<<<grid, block>>>`. *(Day 2)*

**Grid-stride loop** — A launch pattern where a fixed number of threads each process several elements in a loop, striding by the total thread count, instead of sizing the grid to match the data. Correct for any input size without recomputing launch dimensions, and the coalescing-safe way to write **thread coarsening**. *(Day 2)*

**Grid-wide synchronization** — A barrier across every block of a grid, available through cooperative groups, and only for kernels launched with `cudaLaunchCooperativeKernel` and sized so that all blocks are resident at once. *(Day 14)*

**Halo** — The border elements a tile needs but does not own: for a filter of radius R, the R rows and columns around the tile. They must be loaded into shared memory along with the tile. *(Day 5)*

**Hamming distance** — The number of differing bits between two equal-length binary values, computed on GPU via `a ^ b` followed by `__popc`. Used for matching binary feature descriptors such as ORB. *(Day 10)*

**Host** — The CPU, as opposed to the **device** (GPU). *(Day 1)*

**Inclusive and exclusive scan** — Prefix sums. Element i of an inclusive scan is the combination of elements 0 to i; of an exclusive scan, elements 0 to i−1. *(Day 8)*

**Instantiation** — Turning a captured graph into an executable graph with `cudaGraphInstantiate`. Done once; the work of validating and preparing the launches is paid here instead of at every launch. *(Day 12)*

**Instruction pipeline** — The stages an instruction passes through: fetch, decode, register read, execute, memory, writeback. Several instructions are in flight at once, one per stage. *(Day 3)*

**Kernel** — A function marked `__global__`, launched from host code with `<<<grid, block>>>` syntax, executed by many threads in parallel on the device. *(Day 1)*

**Kogge-Stone** — The scan formulation used inside a warp: at step k every lane adds the value from the lane k positions below it, with k doubling each step. Five steps for a 32-lane warp. *(Day 8)*

**L1** — Per-SM cache, physically the same SRAM as shared memory, with a configurable split between the two. *(Day 13)*

**L2** — Chip-wide cache in front of device memory, shared by every SM. Atomics are executed here. *(Day 13)*

**Lane** — A thread's position within its warp, 0 to 31. *(Day 8)*

**Lane mask** — The 32-bit first argument of every `_sync` intrinsic, one bit per lane, naming the lanes that must take part. `0xFFFFFFFF` means the whole warp. *(Day 8)*

**Latency hiding** — The GPU's performance strategy: when one warp stalls, the warp scheduler issues an instruction from a different, ready warp in the same cycle instead of leaving the pipeline idle. The reason GPUs favor many threads over few fast ones. *(Day 3)*

**Launch configuration** — The arguments of a kernel call: grid dimensions in blocks and block dimensions in threads, each up to three-dimensional, plus optional dynamic shared memory size and stream. *(Day 2)*

**Launch overhead** — The host-side cost of issuing one kernel launch. It is what graphs remove, and it matters when a fixed sequence of short kernels runs many times. *(Day 12)*

**Load/store unit** — The SM units that issue memory instructions and compute addresses for global, local and shared memory accesses. *(Day 1)*

**Loop unrolling** — Replacing a loop by repeated copies of its body, which removes branch and index instructions and exposes independent operations to the scheduler. `#pragma unroll` controls it. *(Day 3)*

**LRU** — Least recently used, the eviction order the caches approximate. *(Day 13)*

**Mapped (zero-copy) memory** — Page-locked host memory that also has a device address, from `cudaHostAlloc` with `cudaHostAllocMapped`. A kernel reads and writes it directly across the link with no explicit copy, paying link latency on every access. *(Day 4)*

**Memory bandwidth** — Bytes per second between the SMs and device memory. Theoretical peak is bus width times memory clock times transfers per clock; the achieved figure is what a kernel actually reaches. *(Day 4)*

**Memory pool** — The driver-owned reservation `cudaMallocAsync` allocates from, so a free returns memory to the pool rather than to the OS. Tuned with `cudaMemPoolAttrReleaseThreshold` and inspected through the reserved/used counters. *(Day 15)*

**Mixed precision** — Computing in a narrow format while accumulating in a wider one, typically fp16 or bf16 inputs with fp32 accumulation. This is what tensor cores do natively. *(Day 14)*

**MMA (matrix multiply-accumulate)** — The tensor core operation `D = A * B + C` on small matrix fragments, issued as one instruction per warp. Fragment shapes and alignment are fixed by the hardware, which is why dimensions have to be multiples of the fragment size to use it. *(Day 14)*

**Non-determinism of atomic accumulation** — Atomics do not fix the order in which values are combined, and floating-point addition is not associative, so a kernel accumulating floats with `atomicAdd` can give different results run to run. A reproducibility claim requires a fixed reduction order. *(Day 9, revisited Day 14)*

**nvcc** — The CUDA compiler driver. It separates a `.cu` file into host and device code, compiles the device part itself, passes the host part to the system compiler, and links both into one binary. *(Day 1)*

**NVLink** — NVIDIA's direct GPU-to-GPU link, and on some systems CPU-to-GPU, with several times the bandwidth of PCIe. *(Day 4)*

**N-way conflict** — When N lanes of a warp address different words in the same bank, the access is split into N transactions. For a constant stride N is always `gcd(stride, 32)`, hence always a power of two; odd degrees require an irregular access pattern. *(Day 5)*

**Occupancy** — How many warps are resident on an SM at once relative to the maximum it could hold. Limited by whichever resource runs out first: registers per thread, shared memory per block, or the thread-count cap. It is a means to **latency hiding**, not a goal — returns flatten past roughly 50 percent, and coarsened kernels trade it away deliberately. *(Day 2)*

**Pageable memory** — Ordinary host memory from `malloc` or `new`. The operating system may move or swap its pages, so the GPU cannot access it directly: a transfer first copies it into a page-locked staging buffer held by the driver. *(Day 4)*

**Page-locked memory** — Host memory whose pages the operating system may not move or swap out. *(Day 4)*

**Page migration** — The movement of a unified-memory page to the processor that faulted on it. Repeated migration in both directions is the usual reason unified memory is slow; `cudaMemPrefetchAsync` and `cudaMemAdvise` control it. *(Day 4)*

**PCIe** — The bus connecting host and device on most systems. Its bandwidth is an order of magnitude below device memory bandwidth. *(Day 4)*

**Pinned memory** — Page-locked host memory allocated with `cudaMallocHost`, so the GPU can transfer it by DMA with no staging copy. Required for `cudaMemcpyAsync` to be genuinely asynchronous. OpenCV's equivalent type is `cv::cuda::HostMem`. *(Day 4)*

**Pitch** — The actual byte stride between rows of a 2D allocation (`cudaMallocPitch`, or a `GpuMat`'s `.step`), normally larger than `width * elementSize` because of alignment padding. Kernels touching pitched memory must index rows by pitch, not by width. *(Day 5)*

**`__popc`** — Counts the set bits of a 32-bit value. Applied to a ballot result it counts the lanes that satisfied the predicate; applied to `a ^ b` it gives the Hamming distance. *(Day 8, Day 10)*

**Privatization** — Giving each block, or each warp, a private copy of a contended accumulator, updating that copy locally, and merging once at the end. Turns one global atomic per input element into a few per block. The standard fix for atomic contention. *(Day 9)*

**PTX** — NVIDIA's virtual, forward-compatible GPU assembly language. nvcc compiles device code to PTX first; `ptxas` then assembles PTX into real machine code (**SASS**) for a specific architecture. *(Day 1)*

**Reconvergence** — The point after a divergent branch where all lanes of the warp execute the same instruction again. From Volta on, lanes have independent program counters and reconvergence at the end of a branch is not guaranteed; `__syncwarp` makes it explicit. *(Day 3)*

**Reduction** — Combining N values into one with an associative operator. On a GPU it is done as a tree: within a warp by shuffles, across warps through shared memory, then by one warp again. *(Day 8)*

**Register file** — The per-SM storage from which every thread's registers are allocated. Its size is fixed, so registers per thread and resident warps trade against each other. *(Day 3)*

**Resident blocks** — The blocks assigned to one SM at the same time. The count is the smallest of three limits: the hardware cap on blocks per SM, registers per SM divided by the block's register demand, and shared memory per SM divided by the block's shared memory demand. *(Day 2)*

**Roofline** — A plot of achievable performance against arithmetic intensity: a diagonal bandwidth ceiling that flattens into a horizontal compute ceiling. Which part of the roof a kernel sits under says whether memory or instruction optimizations can help it. *(Day 13)*

**SASS** — The real machine code (cubin) for one specific GPU architecture, assembled from PTX by `ptxas`. *(Day 1)*

**Sector** — The 32-byte unit in which memory is actually requested and moved. A warp's access is counted in sectors, and four sectors make a cache line. Coalescing is the minimization of sectors per request. *(Day 2, measured Day 13)*

**SFU (Special Function Unit)** — The SM units computing transcendental functions — sine, cosine, exponential, reciprocal, reciprocal square root — at lower throughput than the FP32 units. *(Day 1)*

**Shared memory** — On-chip memory allocated per block and shared by its threads, with latency close to a register access and lifetime equal to the block's. Declared `__shared__`, statically or as the dynamic third launch argument. *(Day 5)*

**SIMT (Single Instruction, Multiple Threads)** — NVIDIA's execution model: one instruction is fetched and decoded once and issued to all 32 threads of a warp at the same time. *(Day 3)*

**SM (Streaming Multiprocessor)** — A GPU's core compute unit; a modern GPU has dozens to over a hundred. Each block runs entirely on one SM. Real counts and limits for your GPU are in `report_device_capabilities()`. *(Day 1)*

**Stall reason** — The profiler's classification of why a warp was not eligible: a memory dependency, a barrier, a busy execution pipe, instruction fetch, and so on. It names what to fix. *(Day 3)*

**Stream** — An ordered queue of GPU operations, kernels and copies. Operations in different streams may run concurrently; operations within one stream execute in issue order. *(Day 6)*

**Stream compaction** — Removing the elements that fail a predicate and packing the rest. A scan over the predicate gives each surviving element its output index; for a binary predicate, `__ballot_sync` plus `__popc` is cheaper. *(Day 8)*

**Stream dependency** — Ordering between streams, expressed by recording an event in one and having another wait on it with `cudaStreamWaitEvent`. *(Day 6)*

**Stream-ordered allocation** — `cudaMallocAsync`/`cudaFreeAsync`, which tie allocation and deallocation to a stream instead of forcing the device-wide synchronization classic `cudaMalloc`/`cudaFree` do. Ordered *within one stream* only: using an allocation from another stream needs an explicit event. *(Day 15)*

**Swizzling** — Scrambling a shared-memory index, for example `tile[row][col ^ row]`, so that a fixed logical column maps to a different physical bank on every row, removing bank conflicts without spending a padding column. *(Day 13)*

**`_sync` suffix** — Marks the intrinsics that require the named lanes to be converged at the instruction. If a lane in the mask does not reach it, the result is undefined rather than merely slow. The unsuffixed forms have been removed from the language. *(Day 8)*

**`__syncthreads()`** — A barrier for the block: no thread passes it until every thread of the block reaches it, and shared and global writes made before it are visible to the block after it. Every thread of the block must reach it, so placing it inside divergent control flow is undefined behavior. *(Day 5)*

**`__syncwarp`** — A warp-level barrier forcing the named lanes to converge. Needed where code depends on lanes being together and the compiler cannot prove that they are. *(Day 9)*

**Tensor Core** — Specialized SM hardware, compute capability 7.0 and newer, for fast mixed-precision matrix multiply-accumulate. Used by cuBLAS and cuDNN, and reachable directly through the warp matrix functions. *(Day 1, Day 14)*

**Texture object** — A GPU resource bound to memory (often pitched 2D) that supports hardware-accelerated filtering, for example bilinear, and address clamping on read, accessed in-kernel via `tex2D`. *(Day 11)*

**Theoretical peak bandwidth** — Bus width times memory clock times transfers per clock, computed from the device numbers `report_device_capabilities()` prints on Day 1. The ceiling a kernel is measured against. *(Day 1, applied Day 13)*

**Thread** — The smallest unit of execution; identified within its block by `threadIdx`, within the grid by combining `threadIdx` with `blockIdx` and `blockDim`. *(Day 2)*

**Thread coarsening** — Giving each thread several output elements instead of one, so per-thread fixed costs — index arithmetic, bounds checks, shared-memory tile loads — are paid once and amortized. A grid-stride loop is the coalescing-safe way to write it. It trades against occupancy, so it has to be measured. *(Day 13)*

**Throughput machine and latency machine** — A CPU spends area on caches, branch prediction and out-of-order execution to make one instruction stream fast, that is, to reduce latency. A GPU spends the same area on execution units and register file to keep many warps in flight, and tolerates latency instead of removing it. *(Day 1)*

**Tiling** — Staging a block of data in shared memory once and reading it many times on chip, cutting global traffic by roughly the reuse factor. The technique behind tiled matrix multiply and every stencil and filter kernel in this course. *(Day 5, Day 10, Day 12)*

**Unified memory** — One allocation, `cudaMallocManaged`, addressable from both host and device, with the driver migrating pages between them on demand. *(Day 4)*

**Virtual and real architecture** — `-arch=compute_XX` names the virtual architecture PTX is generated for; `-code=sm_XX` names the real architecture SASS is generated for. `-arch=sm_XX` sets both. *(Day 1)*

**Warp** — A group of 32 threads within a block that the hardware schedules and executes together in lockstep. The unit warp-level intrinsics operate on. *(Day 3, Day 8-9)*

**Warp-aggregated atomics** — Having one lane perform a single atomic for the whole warp's contribution, computed first by a ballot and a warp reduction. Reduces the number of atomics by up to 32 times; the warp-scoped case of privatization. *(Day 8, Day 9)*

**Warp scheduler** — The SM unit that each cycle selects one eligible warp and issues its next instruction to the execution units. An SM has several. *(Day 1)*

**Warp shuffle** — The instruction family `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`, which lets a lane read a register of another lane in the same warp with no memory access and no barrier. *(Day 8)*

**XOR (butterfly) exchange** — `__shfl_xor_sync(mask, v, k)`: lane `i` exchanges with lane `i ^ k`. Every lane both sends and receives in one instruction, which is why it is the form used when all lanes need the result. *(Day 8)*
