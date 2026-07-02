# GPU Architecture Deep Dive: Inside a Streaming Multiprocessor

Day 1 gives you the host-vs-device mental model and the raw numbers for your own GPU via `report_device_capabilities()`. This document goes one level deeper: what's actually *inside* an SM, and how the memory spaces you've been using since Day 4 (pinned/unified), Day 5 (shared/constant/pitched), Day 11 (textures), and Day 13 (L1/L2) map onto real silicon.

Exact unit counts differ by architecture — this describes the general layout every NVIDIA GPU since Volta shares, not a specific chip. Where a number matters, get it from `report_device_capabilities()` for your own GPU rather than trusting a number here.

## Visual
![One SM broken into 4 partitions, each with a warp scheduler, register file segment, INT32/FP32/FP64 units, SFU, tensor core, and load/store units; below the SM, shared memory/L1, constant cache, and texture cache side by side; further below, L2 cache (shared by all SMs, with atomic units) and global memory](sm_anatomy.svg)

## The Compute Units

**Registers / register file.** Each SM has a large, fixed-size register file (`regsPerMultiprocessor` in `report_device_capabilities()`), divided up among all threads currently resident on that SM. A thread's local variables live here — it's the fastest storage that exists, faster than shared memory. The catch: the compiler decides how many registers each thread needs at compile time, and that count multiplied by your block size can't exceed what the SM has. Use too many registers per thread and either fewer blocks fit on the SM at once (occupancy drops), or the compiler starts *spilling* excess variables into **local memory** (see below) — which is actually global memory in disguise, so a spill is a silent, expensive round-trip you didn't ask for. `-Xptxas -v` (Day 1) reports spills; `-maxrregcount` caps registers per thread if you want to trade register pressure for occupancy deliberately.

**ALUs (CUDA cores).** The INT32 arithmetic/logic units — what NVIDIA's marketing calls "CUDA cores." Each SM partition has a bank of these, and a warp's integer instruction (indexing math, comparisons, bitwise ops) is issued across 32 of them at once, one per thread in the warp.

**FPUs — FP32 and FP64.** Separate hardware from the INT32 ALUs. FP32 (single precision) units are usually plentiful — often 1:1 with INT32 cores on recent architectures. FP64 (double precision) units are deliberately scarce on consumer/gaming GPUs, sometimes as few as 1/32nd or 1/64th the FP32 count. That ratio is exactly `prop.singleToDoublePrecisionPerfRatio`, already printed by `report_device_capabilities()` — if it reports 32, your `double` math runs at roughly 1/32 the throughput of the equivalent `float` math on that GPU. This is why `float` is the default choice in this course's kernels unless precision genuinely demands `double`.

**SFU (Special Function Units).** Hardware for fast, lower-precision transcendental math — `sin`, `cos`, `exp`, reciprocal, `sqrt` approximations. What `--use_fast_math` (Day 1) routes your math through instead of the fully IEEE-754-accurate software implementations.

**Tensor Cores.** Matrix-multiply-accumulate hardware, present from Volta onward (compute capability ≥ 7.0). `report_device_capabilities()` reports how many your GPU has per SM; 0 on anything older. Libraries like cuBLAS (Day 14) use these automatically for supported operations and data types.

**Warp scheduler + dispatch unit.** Each SM is split into partitions (commonly 4), each with its own warp scheduler. Every cycle, a scheduler picks one *ready* warp from among all the warps resident in its partition and issues its next instruction to the execution units — this is the hardware behind the latency-hiding story from Day 3: if warp A is stalled waiting on a memory load, the scheduler just issues from warp B instead, so the pipeline doesn't sit idle.

**Load/Store (LD/ST) units.** Handle address calculation and issue for memory instructions — every `cudaMalloc`'d pointer dereference in your kernel goes through these on its way to shared memory, L1, L2, or global memory. Texture fetches (Day 11) go through a separate **texture unit** instead — see the Texture cache entry below.

## From C/C++ to SASS: Instruction Reference

Day 1 explained that nvcc compiles your device code to PTX (virtual assembly), then `ptxas` assembles that into SASS (real machine code) for one specific architecture. This section makes that concrete: what common C/C++ operations actually turn into, and how to see it yourself.

**How to inspect real output for your own kernel** (Day 1's `--keep` flag, taken further):
```bash
nvcc --keep -arch=sm_75 day01_template.cu -o day01   # .ptx lands next to your output
cuobjdump --dump-ptx  day01                           # PTX pulled back out of the binary
cuobjdump --dump-sass day01                           # SASS pulled back out of the binary
nvdisasm day01.cubin                                  # alternative SASS disassembler, if you kept the .cubin
```

**Common operations, roughly Ampere-generation SASS mnemonics (exact names shift slightly by architecture):**

| C/C++ | PTX | SASS (typical) | Notes |
|---|---|---|---|
| `c = a * b + d;` | `fma.rn.f32` | `FFMA` | The compiler *fuses* multiply+add into one instruction automatically — you don't need to ask. Disable with `-fmad=false` if you need strict IEEE separate rounding. |
| `a + b` | `add.f32` | `FADD` | |
| `a * b` | `mul.f32` | `FMUL` | |
| `a / b` | `div.rn.f32` | short instruction sequence, not one opcode | Division isn't a single hardware instruction — it's a reciprocal approximation (via the SFU) refined by a couple of Newton-Raphson steps. This is *why* division is much slower than multiply/add. |
| `sqrtf(a)` | `sqrt.rn.f32` | `MUFU.RSQ` + refinement | Same story as division: approximate via SFU, then refined. `rsqrtf()` skips the refinement for a cheaper, less precise result. |
| `fmodf(a, b)` | no single opcode | `FMUL`/`FFMA`/`FADD` sequence | There's no hardware "modulo" instruction; the compiler expands it into a short sequence (roughly: `a - trunc(a/b) * b`). Worth knowing before you assume it's as cheap as `+`. |
| `__ldg(&x)` (Day 13) | `ld.global.nc.f32` | `LDG.E.CONSTANT` | The `.nc`/`CONSTANT` qualifier routes the read through the read-only data cache path — the same physical cache texture fetches use, see below. |
| shared-memory read/write (Day 5) | `ld.shared` / `st.shared` | `LDS` / `STS` | Distinct opcodes from global loads/stores (`LDG`/`STG`) — the hardware genuinely treats shared memory as a separate address space. |
| `__syncthreads()` | `bar.sync 0` | `BAR.SYNC` | A block-wide barrier instruction — every thread in the block must reach it before any can proceed past it. |
| `__shfl_down_sync(...)` (Day 8) | `shfl.sync.down.b32` | `SHFL.DOWN` | Register-to-register exchange within a warp, no memory traffic at all. |
| `atomicAdd(...)` (Day 9) | `atom.global.add.u32` | `ATOM(G).ADD` / `RED.ADD` | Resolved at L2, not the issuing SM — see the L2 section below. |
| `tex2D(...)` (Day 11) | `tex.2d.v4.f32.f32` | `TEX` | Goes through the dedicated texture unit, not `LDG` — see Texture cache below. |
| `__popc(x)` (Day 10) | `popc.b32` | `POPC` | Population count (number of set bits) in one instruction — what makes Hamming distance cheap. |

**How to actually reach for a specific instruction from C/C++:**

1. **Just write normal arithmetic, most of the time.** `a * b + c` becomes a single `FFMA` automatically. Fighting the compiler to "force" an instruction it would already generate is wasted effort — check the SASS first (via `cuobjdump`) before assuming you need to intervene.

2. **Device math intrinsics, when you need explicit control over rounding or approximation.** CUDA's math API has explicit-rounding-mode versions of the basic ops — `__fadd_rn`, `__fmul_rn`, `__fmaf_rn`, `__dadd_rn`, ... — where `_rn`/`_rz`/`_ru`/`_rd` mean round-to-nearest / toward-zero / up / down. There are also fast, approximate SFU-routed versions — `__expf`, `__logf`, `__sinf`, `__fdividef` — which is what `--use_fast_math` swaps your plain calls for globally, if you'd rather opt individual call sites in instead:
   ```c++
   float precise = __fmaf_rn(a, b, c);      // explicit FMA, round-to-nearest
   float fast    = __fdividef(a, b);        // fast approximate division, via SFU
   ```

3. **Inline PTX assembly, for the rare case with no intrinsic at all.** An escape hatch, not a first resort:
   ```c++
   __device__ int add_via_ptx(int a, int b)
   {
       int result;
       asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
       return result;
   }
   ```
   `%0`/`%1`/`%2` are placeholders bound to the output/input operands listed after the colons; `"r"` means a 32-bit register. You'll see this pattern in some CUDA library source but rarely need it yourself — nearly everything has an intrinsic already.

### Cache Eviction Hints: LRU and Load/Store Cache Operators

Every cache described in this document — L1, L2, texture, constant — is finite, so when it's full and a new line needs to load, something has to be evicted. GPU L1/L2 caches are set-associative (each address maps to one of a small number of "ways" within a "set"), and the hardware replacement policy is an approximation of **LRU (Least Recently Used)**: evict whichever line in the set hasn't been touched in the longest time. It's an *approximation*, not textbook-perfect LRU — tracking exact recency for every line under thousands of concurrent threads is too expensive to build in hardware — but the intent is the same: keep data that's likely to be reused, discard data that isn't.

You can influence this beyond hoping the LRU approximation guesses right, at two different granularities:

**Region-level: `cudaAccessPolicyWindow` (Day 13).** Marks a whole address range as `cudaAccessPropertyPersisting` (bias the policy to *keep* it, resist eviction) or `cudaAccessPropertyStreaming` (bias it to *evict first*). The tool for "this buffer gets read every iteration of my loop — don't let a one-off read evict it."

**Instruction-level: load/store cache operators.** Every individual global load or store can carry its own cache hint, exposed as intrinsics — finer-grained than a whole-buffer policy window, down to a single access:

| Intrinsic | PTX | What actually happens | When to reach for it |
|---|---|---|---|
| `__ldca(ptr)` | `ld.global.ca` | Cached in **both L1 and L2**, subject to the normal approximate-LRU policy — treated exactly like an ordinary dereference. | This is the compiler's default for a plain `*ptr`. Use it explicitly only to make intent obvious in code that's otherwise full of the other hints below. |
| `__ldcg(ptr)` | `ld.global.cg` | Cached **in L2 only — bypasses L1 entirely** on the way in. | You personally won't reuse this value again (so it's not worth L1 space, which is small and per-SM), but a *different* block or SM might read the same address soon — L2 is shared chip-wide, so it still pays off there. |
| `__ldcs(ptr)` | `ld.global.cs` | Cached normally (L1 and L2), but the line is tagged **evict-first** — the replacement policy discards it ahead of ordinary LRU-tracked lines the moment space is needed, even though it was just loaded. | You'll touch this address exactly once, ever, in this kernel. Keeps one-shot data from shouldering out lines that genuinely get reused — e.g. reading a large input array you only sweep through once. |
| `__ldlu(ptr)` | `ld.global.lu` | If the line happens to be in cache, it's **invalidated immediately after this read** — its slot is freed right away rather than aging out under LRU. | You know with certainty this is the *last* read of this address for the rest of the kernel (e.g. the final pass of a multi-stage reduction). Frees cache capacity sooner than `__ldcs` would. |
| `__ldcv(ptr)` | `ld.global.cv` | Treated as **volatile** — bypasses/invalidates any cached copy and re-reads from memory every single time, never trusting what's in L1/L2. | Correctness, not performance: another agent (the host, another stream, another GPU) may have written this address since you last read it — e.g. polling a flag the CPU sets while your kernel is running. |
| `__stwb(ptr, val)` | `st.global.wb` | **Write-back** (default): the value lands in cache, and only gets flushed out to global memory later, when the line is evicted. Repeated writes to the same address before eviction can be absorbed without each one hitting memory. | Ordinary output you might read back later in the same kernel, or that benefits from write-coalescing. |
| `__stcg(ptr, val)` | `st.global.cg` | Write **bypasses L1, lands in L2 only** — mirrors `__ldcg` for stores. | Writing intermediate results you won't re-read yourself, but that another block might soon — avoid burning your own SM's L1 space on it. |
| `__stcs(ptr, val)` | `st.global.cs` | Cached but flagged **evict-first**, same idea as `__ldcs` applied to a store. | Final output written once, never read back inside the kernel — the Day 13 `tiled_filter` output pixel is the textbook example. |
| `__stwt(ptr, val)` | `st.global.wt` | **Write-through**: sent straight to memory (via L2) immediately, instead of sitting dirty in a write-back cache line. | You need the write visible to other kernels/streams/the host as soon as possible, or want to avoid holding a dirty line at all — trades away write-coalescing for earlier visibility. |

This is a *different* mechanism from `__ldg()`: `__ldg` changes **which cache** a read goes through (the read-only/texture cache instead of the normal L1/L2 path — see Day 11 and the Texture cache entry below). The operators above stay on the normal L1/L2 path and instead change **how long the replacement policy tries to keep the line around**. They're complementary, not alternatives — you could `__ldg()` a read-only buffer *and* mark it streaming if you know each element is touched exactly once.

A practical pattern from this week's material: in a kernel like Day 13's `tiled_filter`, the halo region around each tile is read repeatedly by neighboring threads (a good candidate to leave at the default `__ldca` — or route through `__ldg`/texture, Day 11), while the final filtered output is written exactly once per pixel and never read back inside the same kernel — a natural candidate for `__stcs` so it doesn't linger in L1 competing with data that's actually reused.

## Memory Organization

**Shared memory / L1 data cache.** On modern architectures these are the *same* physical on-chip SRAM, split by a configurable ratio (some GPUs let you bias this via `cudaFuncAttributePreferredSharedMemoryCarveout`). Shared memory (Day 5) is what you explicitly manage with `__shared__` arrays; L1 is what automatically caches your kernel's global-memory reads/writes without you asking, using the approximate-LRU replacement policy described above (and steerable per-instruction with the cache operators above). Both live per-SM — not shared across SMs — and both disappear when the block that owns them finishes. Size available via `sharedMemPerBlock` (per-block limit) and the `cudaDevAttrMaxSharedMemoryPerMultiprocessor` attribute (the whole SM's budget, shared across all resident blocks) in `report_device_capabilities()`.

**Constant memory & constant cache.** A small (typically 64 KB total, `totalConstMem`), read-only memory space with its own small per-SM cache. Its performance characteristic is unusual: if every thread in a warp reads the *same* address, the cache broadcasts that one value to all 32 threads in a single cycle — full speed. If threads read *different* addresses, those reads serialize. That's why constant memory is for genuinely constant, uniformly-read data (a convolution kernel's coefficients, a small lookup table), not per-thread data.

**Kernel parameters live in constant memory.** This one surprises people: the arguments you pass to a `__global__` function (`my_kernel<<<grid,block>>>(a, b, n)`) aren't passed on a stack or in registers the way a normal C++ function call works — the driver copies them into a reserved region of constant memory before the kernel launches, and every thread reads them from there. This is exactly the constant-cache broadcast case above: every thread in a warp reading the same kernel parameter is effectively free.

**Texture cache.** A third small, read-only, per-SM cache, distinct from both L1 and constant cache — this is what Day 11's `tex2D` fetches actually hit. On Kepler and newer, it's unified with the same "read-only data cache" path that `__ldg()` uses (see the instruction table above), so a plain `__ldg()`'d pointer and an actual bound texture object can end up sharing the same physical cache. What earns it a separate name from L1: it's tuned for **2D/3D spatial locality** rather than linear coalescing. A fetch at `(x, y)` and a nearby fetch at `(x+1, y+1)` hit this cache well even though those two addresses aren't contiguous in linear memory — exactly the access pattern Day 11's zoom/rotate kernels have, and precisely what a cache built for coalesced linear access (L1) doesn't handle as gracefully. The **texture unit** sitting in front of this cache is also where the bilinear filtering and address-mode (clamp/wrap/mirror) hardware from Day 11 physically lives — the cache alone doesn't interpolate anything; the unit does that on the way out.

**L2 cache & atomics.** Unlike shared memory/L1/constant/texture caches, L2 is a single cache shared by *every* SM on the chip, sitting between all the SMs and global memory (`l2CacheSize` in `report_device_capabilities()`; `persistingL2CacheMaxSize` is the portion you can pin with the `cudaAccessPolicyWindow` hints from Day 13, and it uses the same approximate-LRU replacement policy — biased by that policy window — as L1). L2 also contains dedicated ALUs for atomic read-modify-write operations — when you call `atomicAdd` (Day 9), the operation is actually resolved at L2, not bounced back to the issuing SM's own ALUs. That's *why* heavy atomic contention on a single address is slow: every SM's atomic requests to that address funnel through the same L2 slice and serialize there, regardless of how many SMs are trying.

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

Texture cache size specifically isn't exposed through `cudaDeviceProp` the way the others are — NVIDIA doesn't publish it as a queryable attribute, so there's no row for it here.

## Where This Connects in the Course
- **Day 1** — `report_device_capabilities()` surfaces the raw numbers this document explains; `--keep` and `-Xptxas -v` are how you inspect the PTX/SASS discussed above.
- **Day 3** — warp scheduling and the instruction pipeline (fetch from I-cache, dispatch to a partition) are the *behavior*; this document is the *hardware* behind it.
- **Day 4** — pinned/unified memory is about the host↔device link; this document is what's on the far side of that link, inside the device.
- **Day 5, Day 13** — shared memory, bank conflicts, `__ldg`, LRU/cache-operator hints, and L2 persistence hints are all techniques for working *with* the memory organization described here, not around it.
- **Day 8, Day 9, Day 10** — `__shfl_*`, `__ballot_sync`, `__popc`, and `atomicAdd` are all single instructions once you get past the intrinsic wrapper — the instruction table above shows what they actually compile to.
- **Day 11** — texture sampling and bilinear filtering are backed by the texture cache and texture unit described above, not by L1.
