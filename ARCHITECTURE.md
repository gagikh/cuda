# GPU Architecture Deep Dive: Inside a Streaming Multiprocessor

Day 1 gives you the host-vs-device mental model and the raw numbers for your own GPU via `report_device_capabilities()`. This document goes one level deeper: what's actually *inside* an SM, and how the memory spaces you've been using since Day 4 (pinned/unified), Day 5 (shared/constant/pitched), and Day 13 (L1/L2) map onto real silicon.

Exact unit counts differ by architecture — this describes the general layout every NVIDIA GPU since Volta shares, not a specific chip. Where a number matters, get it from `report_device_capabilities()` for your own GPU rather than trusting a number here.

## Visual
![One SM broken into 4 partitions, each with a warp scheduler, register file segment, INT32/FP32/FP64 units, SFU, tensor core, and load/store units; below the SM, shared memory/L1 and constant cache; further below, L2 cache (shared by all SMs, with atomic units) and global memory](sm_anatomy.svg)

## The Compute Units

**Registers / register file.** Each SM has a large, fixed-size register file (`regsPerMultiprocessor` in `report_device_capabilities()`), divided up among all threads currently resident on that SM. A thread's local variables live here — it's the fastest storage that exists, faster than shared memory. The catch: the compiler decides how many registers each thread needs at compile time, and that count multiplied by your block size can't exceed what the SM has. Use too many registers per thread and either fewer blocks fit on the SM at once (occupancy drops), or the compiler starts *spilling* excess variables into **local memory** (see below) — which is actually global memory in disguise, so a spill is a silent, expensive round-trip you didn't ask for. `-Xptxas -v` (Day 1) reports spills; `-maxrregcount` caps registers per thread if you want to trade register pressure for occupancy deliberately.

**ALUs (CUDA cores).** The INT32 arithmetic/logic units — what NVIDIA's marketing calls "CUDA cores." Each SM partition has a bank of these, and a warp's integer instruction (indexing math, comparisons, bitwise ops) is issued across 32 of them at once, one per thread in the warp.

**FPUs — FP32 and FP64.** Separate hardware from the INT32 ALUs. FP32 (single precision) units are usually plentiful — often 1:1 with INT32 cores on recent architectures. FP64 (double precision) units are deliberately scarce on consumer/gaming GPUs, sometimes as few as 1/32nd or 1/64th the FP32 count. That ratio is exactly `prop.singleToDoublePrecisionPerfRatio`, already printed by `report_device_capabilities()` — if it reports 32, your `double` math runs at roughly 1/32 the throughput of the equivalent `float` math on that GPU. This is why `float` is the default choice in this course's kernels unless precision genuinely demands `double`.

**SFU (Special Function Units).** Hardware for fast, lower-precision transcendental math — `sin`, `cos`, `exp`, reciprocal, `sqrt` approximations. What `--use_fast_math` (Day 1) routes your math through instead of the fully IEEE-754-accurate software implementations.

**Tensor Cores.** Matrix-multiply-accumulate hardware, present from Volta onward (compute capability ≥ 7.0). `report_device_capabilities()` reports how many your GPU has per SM; 0 on anything older. Libraries like cuBLAS (Day 14) use these automatically for supported operations and data types.

**Warp scheduler + dispatch unit.** Each SM is split into partitions (commonly 4), each with its own warp scheduler. Every cycle, a scheduler picks one *ready* warp from among all the warps resident in its partition and issues its next instruction to the execution units — this is the hardware behind the latency-hiding story from Day 3: if warp A is stalled waiting on a memory load, the scheduler just issues from warp B instead, so the pipeline doesn't sit idle.

**Load/Store (LD/ST) units.** Handle address calculation and issue for memory instructions — every `cudaMalloc`'d pointer dereference in your kernel goes through these on its way to shared memory, L1, L2, or global memory.

## Memory Organization

**Shared memory / L1 data cache.** On modern architectures these are the *same* physical on-chip SRAM, split by a configurable ratio (some GPUs let you bias this via `cudaFuncAttributePreferredSharedMemoryCarveout`). Shared memory (Day 5) is what you explicitly manage with `__shared__` arrays; L1 is what automatically caches your kernel's global-memory reads/writes without you asking. Both live per-SM — not shared across SMs — and both disappear when the block that owns them finishes. Size available via `sharedMemPerBlock` (per-block limit) and the `cudaDevAttrMaxSharedMemoryPerMultiprocessor` attribute (the whole SM's budget, shared across all resident blocks) in `report_device_capabilities()`.

**Constant memory & constant cache.** A small (typically 64 KB total, `totalConstMem`), read-only memory space with its own small per-SM cache. Its performance characteristic is unusual: if every thread in a warp reads the *same* address, the cache broadcasts that one value to all 32 threads in a single cycle — full speed. If threads read *different* addresses, those reads serialize. That's why constant memory is for genuinely constant, uniformly-read data (a convolution kernel's coefficients, a small lookup table), not per-thread data.

**Kernel parameters live in constant memory.** This one surprises people: the arguments you pass to a `__global__` function (`my_kernel<<<grid,block>>>(a, b, n)`) aren't passed on a stack or in registers the way a normal C++ function call works — the driver copies them into a reserved region of constant memory before the kernel launches, and every thread reads them from there. This is exactly the constant-cache broadcast case above: every thread in a warp reading the same kernel parameter is effectively free.

**L2 cache & atomics.** Unlike shared memory/L1, L2 is a single cache shared by *every* SM on the chip, sitting between all the SMs and global memory (`l2CacheSize` in `report_device_capabilities()`; `persistingL2CacheMaxSize` is the portion you can pin with the `cudaAccessPolicyWindow` hints from Day 13). L2 also contains dedicated ALUs for atomic read-modify-write operations — when you call `atomicAdd` (Day 9), the operation is actually resolved at L2, not bounced back to the issuing SM's own ALUs. That's *why* heavy atomic contention on a single address is slow: every SM's atomic requests to that address funnel through the same L2 slice and serialize there, regardless of how many SMs are trying.

**Global memory (VRAM).** Off-chip DRAM, the largest and slowest space, visible to every SM through L2. Everything you `cudaMalloc` lives here. This is the "Device" memory in Day 1's host/device picture.

**Local memory.** Despite the name, this is *not* on-chip — it's a per-thread private region carved out of global memory, used automatically for register spills and large local arrays the compiler can't fit in registers. It's cached through L1/L2 like any other global memory access, but from the programmer's side it behaves like a normal local variable. A kernel that looks correct but is mysteriously slow is worth checking with `-Xptxas -v` for local-memory usage (spills).

**Instruction cache.** Where the SASS code for your kernel (Day 1's PTX → SASS pipeline) actually sits while an SM executes it — a small, fast I-cache per SM (sometimes per partition), backed by a larger shared instruction cache. This is what the *Fetch* stage in Day 3's pipeline diagram reads from.

## How This Maps to `report_device_capabilities()`

| Field | What it tells you |
|---|---|
| `regsPerMultiprocessor` / `regsPerBlock` | Register file size — the budget behind register spilling and occupancy |
| `sharedMemPerBlock` / max shared mem per SM | Shared memory / L1 budget (Day 5, Day 13) |
| `totalConstMem` | Constant memory size (and indirectly, headroom for kernel parameters) |
| `l2CacheSize` / `persistingL2CacheMaxSize` | L2 cache size and how much of it you can pin (Day 13) |
| `warpSize` | Threads per warp — almost always 32, never hardcode it anyway |
| `maxThreadsPerMultiProcessor` | Ceiling on resident warps per SM — the other half of the occupancy equation |
| `singleToDoublePrecisionPerfRatio` | How many FP32 units exist per FP64 unit |
| tensor cores per SM | Whether/how much cuBLAS-style matrix hardware you have (Day 14) |
| `memoryClockRate` / `memoryBusWidth` | Theoretical global-memory bandwidth — the ceiling nothing beats |

## Where This Connects in the Course
- **Day 1** — `report_device_capabilities()` surfaces the raw numbers this document explains.
- **Day 3** — warp scheduling and the instruction pipeline (fetch from I-cache, dispatch to a partition) are the *behavior*; this document is the *hardware* behind it.
- **Day 4** — pinned/unified memory is about the host↔device link; this document is what's on the far side of that link, inside the device.
- **Day 5, Day 13** — shared memory, bank conflicts, `__ldg`, and L2 persistence hints are all techniques for working *with* the memory organization described here, not around it.
- **Day 9** — `atomicAdd` contention costs make a lot more sense once you know atomics resolve at L2, not at the SM.
