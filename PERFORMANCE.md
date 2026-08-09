# The Performance Checklist

[ARCHITECTURE.md](ARCHITECTURE.md) describes what the hardware *is*. This document is about what to *do* with it: the handful of techniques that account for nearly all real CUDA speedups, in the order you should try them, and how to know when to stop.

The techniques below are not a menu to pick from at random. They target different bottlenecks, and applying the wrong one is at best wasted effort — at worst it makes things slower. So the method comes first.

---

## 0. The method: measure, then decide

The single most common mistake in GPU optimization is optimizing the thing that isn't the bottleneck. Two numbers prevent it.

**Theoretical peak bandwidth**, printed by `report_device_capabilities()` on Day 1:

```c++
const auto memory_clock_rate_mhz = prop.memoryClockRate / 1000.0;
const auto bus_width_bytes       = prop.memoryBusWidth / 8;
const auto bandwidth_gb_s        = 2.0 * memory_clock_rate_mhz * bus_width_bytes / 1000.0;
```

Every term in that line earns its place, and each one is a unit conversion people routinely get wrong:

| Term | Why |
|---|---|
| `prop.memoryClockRate` | **The unit is kilohertz.** Not Hz, not MHz — the CUDA runtime reports all clocks in kHz. An RTX 4090 returns `10501000`. |
| `/ 1000.0` | kHz → MHz, i.e. millions of cycles per second. `10501000 kHz` → `10501 MHz`. |
| `prop.memoryBusWidth` | **The unit is bits** — how many bits move across the bus per transfer. 384 on a 4090; 5120 on an A100, because HBM is very wide and comparatively slow where GDDR is narrow and fast. |
| `/ 8` | bits → bytes, because bandwidth is quoted in bytes. 384 bits → 48 bytes per transfer. Integer division is safe here: every real bus width (64, 128, 192, 256, 384, 512, 5120) is a multiple of 8. |
| `2.0 *` | **DDR — Double Data Rate.** GDDR and HBM both transfer on the rising *and* the falling edge of every clock cycle, so one cycle moves two bus-widths of data. This 2 is a property of the memory technology. It is **not** the same 2 as in the peak-FLOPs formula below, where the 2 is an FMA counting as two operations — unrelated things that happen to share a constant. |
| `/ 1000.0` | `MHz × bytes` is already MB/s (10⁶ transfers/s × bytes each), so one more factor of 1000 gives GB/s. |

Worked end to end for a 4090:

```
10501000 kHz / 1000   = 10501 MHz
384 bits / 8          = 48 bytes per transfer
2 × 10501 × 48        = 1,008,096 MB/s
/ 1000                = 1008 GB/s          <- matches the published spec
```

The same formula checks out against A100 (1935 GB/s), RTX 3090 (936), V100 (900) and H100 SXM (3350). Note that GB here means 10⁹ bytes, not 2³⁰ — the decimal convention vendors quote bandwidth in.

One caveat that changes how you read the ratio: this is a **theoretical ceiling**, what the bus would sustain if it never idled. Real kernels top out around 80–90% even when perfectly coalesced, because of DRAM refresh, row activation and read/write turnaround. So 85% of peak means you're done, not that there's 15% left on the table.

**Achieved bandwidth**, which you compute from a kernel's timing (Day 6's `cudaEvent`s):

```
achieved_GB_s = bytes_read_and_written / elapsed_seconds / 1e9
```

Divide the second by the first. That percentage is the only optimization metric that matters at the start:

| % of peak | What it means | What to do |
|---|---|---|
| < 30% | Something is structurally wrong | Check coalescing (§1) and occupancy (§2) before anything else |
| 50–70% | Normal for a first correct kernel | Work the checklist below in order |
| > 80% | You are near the hardware limit | **Stop.** No amount of cleverness beats the memory bus. The only remaining win is doing less I/O (§7) |

A vector add reads 2 floats and writes 1 per element — 12 bytes of traffic for one add. It will *never* be compute-bound, and no amount of loop unrolling or fast math will help it. Knowing that before you start saves a day.

> **Try it now:** Day 13's `template.cu` already times three kernel variants. Add the bandwidth calculation to each `printf` and you have converted "this one is faster" into "this one hits 78% of peak, so stop."

[`common/timer.h`](common/timer.h) packages all of this — `cudaEvent` timing with running avg/min/max, plus `gb_per_s()`, `tflops()` and the two peaks to divide by:

```c++
#include "../common/timer.h"

kernel_timer_t t;
for (int i = 0; i < 100; ++i) {     // one measurement is mostly noise
    t.start();
    my_kernel<<<grid, block>>>(...);
    t.stop();
}
t.report_bandwidth("my_kernel", 3.0 * n * sizeof(float));   // read a, read b, write c
```
```
my_kernel                   0.412 ms     763.1 GB/s  (76% of 1008 GB/s peak)
```

### A note on the profilers

The arithmetic above tells you *whether* a kernel is slow. When you need to know *why*, there are two tools, and picking the wrong one wastes an afternoon:

- **`nsys` (Nsight Systems) — timeline, whole application.** Answers "where does wall-clock time actually go?" Use it when the GPU looks idle, when you suspect transfers dominate compute (Days 4, 6, 7), or when you want to confirm streams are genuinely overlapping rather than serializing. `nsys profile -o report ./day07`, then open `report.nsys-rep` in the GUI. The give-away pattern is a timeline with gaps between kernels: your bottleneck is host-side or a missing async copy, and no amount of kernel tuning will help.
- **`ncu` (Nsight Compute) — one kernel, in depth.** Answers "why is *this* kernel slow?" Use it once `nsys` has told you which kernel matters. `ncu --set full -o report ./day13` is thorough and slow; targeted metrics are faster:

```bash
ncu --metrics gpu__time_duration.sum,\
dram__bytes.sum.per_second,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./day13
```

Those last two are the whole memory-bound-vs-compute-bound question in two numbers: whichever percentage is higher is your bottleneck, and if neither is above ~60% you're latency-bound (not enough work in flight — look at occupancy, §2). Two more worth knowing:

```bash
# coalescing (§1): sectors per request. 4 is perfect for float, ~32 means every lane its own line
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./prog
# shared-memory bank conflicts (Day 5). Should be 0
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./prog
```

`ncu` also has a **Speed of Light** section that reports achieved vs. peak for compute and memory directly — the same ratio as §0, computed for you. If you only ever read one thing in the profiler, read that.

Two practical notes: `ncu` needs elevated permissions on many systems (`--target-processes all`, or the `NVreg_RestrictProfilingToAdminUsers=0` driver option on Linux), and always compile with `-lineinfo` (Day 1) so the profiler can attribute stalls to source lines.

### The other half: TFLOP/s

Bandwidth is the right metric for a memory-bound kernel. For a compute-bound one — matmul, convolution, anything with real arithmetic per byte — the metric is **TFLOP/s**, and the shape is the same: count the useful operations, divide by time, compare against peak.

**Counting the FLOPs.** For an M×M×M matrix multiply, each of the M² output elements is a dot product of length M: M multiplies and M adds. So:

```
FLOPs = 2 * M^3
```

The factor of 2 is the multiply *and* the add. Counting an FMA as one operation instead of two is the most common way people accidentally report half their real throughput — the convention everyone else uses (and that cuBLAS is benchmarked with) is 2.

**Converting to TFLOP/s**, where the timing is in milliseconds:

```c++
/**
 * @brief Achieved TFLOP/s for an M x M x M matrix multiply.
 * @param M Matrix dimension
 */
auto tflops(int M) const
{
    const double avg_ms = static_cast<double>(totalTime_) / count_;
    return 2.0 * M * M * M * 1e-9 / avg_ms;
}
```

Where `1e-9` comes from: FLOPs ÷ (ms/1000) ÷ 1e12 collapses to FLOPs × 1e-9 ÷ ms. Sanity check it once — a 4096³ matmul in 10 ms is 2·4096³ = 1.374×10¹¹ FLOPs in 0.01 s = **13.7 TFLOP/s**, and the formula gives 1.374×10¹¹ × 1e-9 / 10 = 13.7. ✓

Two things to be careful about in that snippet:

- **Do the division in floating point.** If `totalTime_` and `count_` are integer types, `totalTime_ / count_` truncates *before* you cast, and a sub-millisecond kernel reports `0` or `inf`. Cast first, divide second — as above.
- **`M * M * M` overflows `int` at M = 1291.** Promote to `double` (or `size_t`) before the multiply, not after. Writing `2.0 * M * M * M` forces the promotion on the first operation; `2 * M * M * M * 1e-9` does the whole cube in `int` first and silently wraps.

**Peak FP32 for comparison**, from numbers `report_device_capabilities()` already prints:

```
peak_TFLOPs = 2 * num_SMs * fp32_lanes_per_SM * clock_GHz / 1000
```

The leading 2 is again the FMA (one instruction, two FLOPs). `fp32_lanes_per_SM` is 64 on A100-class parts and 128 on consumer Ampere/Ada — the CUDA-core split from [ARCHITECTURE.md](ARCHITECTURE.md), and the one figure not directly queryable. For an RTX 4090: `2 × 128 SMs × 128 lanes × 2.52 GHz / 1000 ≈ 82.6 TFLOP/s`, which matches the spec sheet.

Then, exactly as with bandwidth, the ratio is what you act on. A hand-written tiled matmul at 8 TFLOP/s on an 82 TFLOP/s card is at 10% of peak — cuBLAS will be 5–8× faster and you should use it. The same kernel at 60% of peak means you're competitive and the remaining gap is tensor cores, not tiling.

> **Which metric applies?** Compute the machine's **balance point**: peak FLOP/s ÷ peak bandwidth. On a 4090 that's 82.6e12 / 1008e9 ≈ **82 FLOP/byte**. If your kernel's arithmetic intensity is below that, quote GB/s; above it, quote TFLOP/s. Quoting TFLOP/s for a vector add (0.083 FLOP/byte) is meaningless — it will always look terrible, and the number you should be reporting is bandwidth, where it may well be at 95% of peak.

### Arithmetic intensity and the roofline

Formalize the same idea by computing a kernel's **arithmetic intensity**: FLOPs performed ÷ bytes moved.

```
vector add     c[i] = a[i] + b[i]      1 FLOP / 12 bytes  = 0.083  -> hopelessly memory-bound
SAXPY          y[i] = a*x[i] + y[i]    2 FLOP / 12 bytes  = 0.17   -> memory-bound
naive matmul   N^3 mults, N^3*2 loads  0.25 FLOP/byte             -> memory-bound
tiled matmul   same math, N^3*2/T loads   ~T/8 FLOP/byte          -> compute-bound once T is large
```

Plot achieved GFLOP/s against arithmetic intensity and you get the **roofline**: a diagonal line (bandwidth × intensity — the memory ceiling) that flattens into a horizontal line (peak FLOP/s — the compute ceiling). Every kernel sits under it. Which part of the roof you're under tells you which optimizations can possibly help:

- **Under the slope (memory-bound):** you are limited by bytes. Coalescing (§1), thread coarsening (§6), and reducing total traffic (§7) help. Faster math does nothing.
- **Under the flat part (compute-bound):** you are limited by instructions. Occupancy (§2), divergence (§3), and cheaper math help. Better memory access does nothing.

The crossover point is a property of the GPU, not your kernel — it's peak FLOP/s ÷ peak bandwidth, typically somewhere around 10–100 FLOP/byte on modern hardware. Note how high that is: **most kernels most people write are memory-bound**, which is why this course spends so much of Days 4–13 on memory.

Notice that tiled matmul is the same arithmetic as naive matmul; tiling didn't make the math faster, it moved the kernel from one part of the roofline to the other by cutting bytes. That's the general shape of a good optimization.

---

## 1. Coalescing

**Get this right before anything else.** *Covered on [Day 2](day02/README.md#memory-coalescing) (the pattern) and [Day 13](day13/README.md) (the measurement).*

When a warp issues a global load, the memory system serves it in **128-byte transactions**. If the 32 lanes ask for 32 consecutive 4-byte words, those requests fall inside one transaction and the warp is served once. If they're scattered, the hardware issues a separate transaction per distinct line — up to 32 of them — fetching 128 bytes each to use 4.

That's not 32× more instructions. It's up to **32× more bytes off the bus**, on the resource that is already your ceiling. Nothing else on this list has that leverage, which is why it goes first.

```c++
// COALESCED: adjacent lanes touch adjacent addresses
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = in[i];

// NOT COALESCED: each lane starts a whole row apart
int i = threadIdx.x * n + blockIdx.x;
out[i] = in[i];
```

Both are correct. Both do the same work. On a large array the second can be an order of magnitude slower.

The rule is about **the warp, not the thread**. A single thread walking memory sequentially in a loop is the *bad* pattern; 32 threads each grabbing one adjacent element is the good one. That inverts the intuition you brought from CPU code, where sequential access per core is exactly what you want — this is the single most common transfer error from CPU to GPU programming.

**Checklist:**
- Does `threadIdx.x` appear in the *fastest-varying* part of the index expression? If it's multiplied by anything, you have a stride.
- For 2D data, is `threadIdx.x` mapped to the **column** (contiguous) and `threadIdx.y` to the row? Swapping them is a silent 10× loss.
- Struct-of-arrays beats array-of-structs. `struct {float x,y,z;} p[N]` makes `p[i].x` a stride-3 access; three separate `x[N]`, `y[N]`, `z[N]` arrays are all stride-1.
- If the access is genuinely irregular (a gather), coalescing isn't available — stage through shared memory instead, which is where §5 comes in.

**Measure it:** `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./your_kernel`. Sectors ÷ requests tells you how many 32-byte sectors each request pulled. Perfectly coalesced float access gives 4; a value near 32 means every lane is fetching its own line.

---

## 2. Occupancy

**Necessary, not sufficient.** *Introduced on [Day 2](day02/README.md#occupancy).*

**Occupancy** is the ratio of warps actually resident on an SM to the maximum it could hold. It matters because latency hiding (Day 3) is the GPU's entire performance strategy: when one warp stalls on a memory load, the scheduler needs *another ready warp* to issue from. No spare warps, no hiding, and the SM sits idle through every memory latency.

Three resources cap it, and the tightest one wins:

| Limiter | How to check | How to relieve it |
|---|---|---|
| **Registers per thread** | `nvcc -Xptxas -v` (Day 1) | `-maxrregcount`, `__launch_bounds__`, or simplify the kernel |
| **Shared memory per block** | Your own `__shared__` declarations | Smaller tiles, or fewer bytes per element |
| **Block size** | Your launch config | Blocks that aren't a multiple of 32 waste part of a warp |

Worked example. An SM allows 2048 resident threads, 64K registers and 96 KB of shared memory. Your kernel uses 40 registers/thread and 8 KB of shared memory per 256-thread block:

```
registers:     65536 / (40 * 256)  = 6.4  -> 6 blocks
shared memory: 98304 / 8192        = 12   -> 12 blocks
thread limit:  2048  / 256         = 8    -> 8 blocks
                                            ^ registers win: 6 blocks
occupancy = 6 blocks * 256 threads / 2048 = 75%
```

Registers are the binding constraint, so shrinking the tile would buy nothing — only cutting register pressure moves the number. That's the point of computing it rather than guessing.

Let the runtime do this for you:

```c++
int blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, my_kernel, 0, 0);

int maxBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, my_kernel, blockSize, sharedMemBytes);
float occupancy = (maxBlocks * blockSize / 32.0f) / (prop.maxThreadsPerMultiProcessor / 32.0f);
```

**The caveat that matters more than the metric.** Occupancy is a means, not a goal. Past roughly 50% the returns flatten hard, and a *lower*-occupancy kernel that gives each thread more registers and more independent work in flight is often faster — this is exactly the thread-coarsening trade in §6. Volkov's well-known result is that the fastest matmul kernels run at low occupancy on purpose. Treat below ~30% as a red flag worth investigating; treat 60% vs 75% as noise.

---

## 3. Control divergence

*Covered on [Day 3](day03/README.md) — the one item on this checklist the course already handles well.*

A warp executes one instruction across 32 lanes. When lanes disagree on a branch, the hardware runs both paths with the inactive lanes masked off, so the warp pays for both.

The cost depends entirely on how divergence lines up with warp boundaries:

```c++
if (threadIdx.x % 2 == 0) { A(); } else { B(); }   // every warp diverges: pays A + B
if (threadIdx.x / 32 % 2 == 0) { A(); } else { B(); } // whole warps take one path: free
```

Same branch, same amount of work, completely different cost. Where the condition comes from data rather than thread index you can sometimes sort or bucket the data so that similar work lands in the same warp.

Watch out for the non-obvious sources: an early `return` on a bounds check, a `while` loop with a data-dependent trip count (the whole warp runs until the *last* lane finishes), and `switch` on a per-thread value.

---

## 4. Tiling and shared memory

*Covered on [Day 5](day05/README.md), [Day 10](day10/README.md), [Day 12](day12/README.md) — the other strength.*

Stage a block of data in shared memory once, then read it many times from on-chip. The win is a division: if every element is reused `T` times, tiling cuts global traffic by roughly `T`.

Tiled matmul is the canonical case — naive matmul reads each element of A and B N times from global; a `T×T` tiled version reads each `N/T` times. The arithmetic is identical. That's the roofline move from §0.

Two follow-ons the course covers: bank conflicts ([Day 5](day05/README.md), and the configurable [bank_conflict_animations.html](day05/bank_conflict_animations.html)), and the padding vs. swizzling trade ([Day 13](day13/README.md)).

---

## 5. Privatization

*See [Day 9](day09/README.md#privatization).*

When many threads atomically update the *same* address, they serialize — atomics resolve at L2 (see [ARCHITECTURE.md](ARCHITECTURE.md)), so every SM's request to that address funnels through one L2 slice regardless of how much parallelism you have. A histogram over 256 bins with a million threads is the standard disaster.

**Privatization** gives each block (or each warp) its own private copy, updates that, and merges once at the end:

```c++
__global__ void histogram_private(const unsigned char *in, int n, unsigned int *bins)
{
    __shared__ unsigned int local[256];

    // 1. zero the private copy
    for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
    __syncthreads();

    // 2. accumulate into shared memory -- contention is now within one block,
    //    and shared-memory atomics are far cheaper than global ones
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) atomicAdd(&local[in[i]], 1);
    __syncthreads();

    // 3. one merge per block: 256 global atomics instead of n
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (local[i]) atomicAdd(&bins[i], local[i]);
    }
}
```

The accounting is the whole story: `n` global atomics become `n` *shared* atomics plus `256 × numBlocks` global ones. For n = 10⁷ and 1000 blocks that's 10 million global atomics down to 256 thousand — a 40× reduction in traffic to the contended resource.

**Two cheaper variants, same idea:**

- **Warp-level aggregation.** Reduce within the warp first, then have lane 0 issue one atomic — 32× fewer requests. This is exactly what Day 8 and Day 9 already build with `__shfl_down_sync`; privatization is that pattern generalized past sum reduction. `__match_any_sync` ([INTRINSICS.md](INTRINSICS.md)) does it for the keyed case.
- **Block-scoped atomics.** `atomicAdd_block()` only needs coherence within the block, so it's cheaper than the device-wide version. Free win on shared-memory accumulators.

**When it doesn't pay:** if contention is already low (a large output array, well-spread keys), privatization just adds a merge phase and zeroing cost. It's a fix for *contention*, so confirm you have contention before reaching for it.

---

## 6. Thread coarsening

*See [Day 13](day13/README.md#thread-coarsening).*

The default instinct is one thread per output element. Sometimes that's too much parallelism: every thread re-pays a fixed cost — index arithmetic, bounds checks, the tile loads it shares with its neighbours, block launch overhead. **Coarsening** gives each thread several elements so that cost is paid once and amortized.

```c++
// one element per thread
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = f(in[i]);

// four elements per thread: index math and loop setup paid once
int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
#pragma unroll
for (int k = 0; k < 4; ++k) if (i + k < n) out[i + k] = f(in[i + k]);
```

Careful — **that version breaks coalescing.** Thread 0 takes 0–3, thread 1 takes 4–7, so lanes are now stride-4. The correct form keeps the grid-stride shape from Day 2, so lanes stay adjacent on every iteration:

```c++
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int k = i; k < n; k += stride) out[k] = f(in[k]);   // still coalesced
```

Which is a nice result: **the grid-stride loop Day 2 already teaches is thread coarsening**, and it's the coalescing-safe way to write it. The knob is the grid size — launch fewer blocks and each thread naturally handles more elements.

The bigger win is in tiled kernels, where a coarsened thread computes several outputs from *one* set of shared-memory loads:

```
tiled matmul, 1 output/thread : load a tile of A and a tile of B -> 1 result
tiled matmul, 4 outputs/thread: load a tile of A and a tile of B -> 4 results
```

Same loads, four times the useful work — arithmetic intensity up 4×, which is exactly the §0 roofline move again.

**The trade-off:** more registers per thread (each partial result lives in one), so occupancy drops. That's usually fine — see the §2 caveat — but it means coarsening and occupancy pull in opposite directions and you have to measure rather than reason. Typical sweet spot is 2–8 elements per thread; past that, register spilling to local memory eats the gain, and `-Xptxas -v` will tell you when you've crossed it.

---

## 7. Rewriting the algorithm with better math

The techniques above make a given algorithm run closer to the hardware limit. This one moves the limit, and it is by far the largest lever available — but it requires actually thinking about the problem rather than the code.

**Reduce the bytes.**
- *Lower precision.* FP16 or BF16 halves every byte moved and unlocks tensor cores. INT8 quarters it. For a memory-bound kernel that's a direct 2–4× — the reason quantization dominates ML inference.
- *Fuse kernels.* Three elementwise kernels each read and write the whole array: 6 passes over memory. Fused into one: 2 passes. Nothing about the arithmetic changed.
- *Recompute instead of storing.* Counter-intuitive on a CPU, routine on a GPU: FLOPs are nearly free and bytes are not, so recomputing a cheap intermediate often beats a round trip to global memory.

**Change the algorithm's complexity or its memory pattern.**
- *Better asymptotics.* A work-efficient Brent-Kung scan does O(n) work where the simpler Kogge-Stone does O(n log n) — see PMPP Ch. 11.
- *Restructure to avoid materializing anything large.* **FlashAttention** is the famous example: standard attention writes an N×N score matrix to global memory and reads it back, so it's memory-bound and O(N²) in memory. FlashAttention tiles the computation and uses an online softmax so the score matrix never exists in HBM at all. The FLOP count actually goes *up* slightly. It's several times faster, because it traded free FLOPs for expensive bytes.

**Use a library.** cuBLAS, cuFFT, CUB and Thrust ([Day 14](day14/README.md)) are written by people with access to the SASS scheduler and years of tuning per architecture. If your problem is a GEMM, you will not beat cuBLAS. Reach for a library first; write a custom kernel when your problem *isn't* the library's problem — usually because fusing it with neighbouring work saves a memory round trip, which is §7 again.

---

## Putting it in order

1. **Is it correct?** `compute-sanitizer` (Day 1) before anything else. A fast wrong kernel is worth nothing.
2. **What fraction of peak are you at?** (§0) If > 80%, stop.
3. **Are the accesses coalesced?** (§1) Biggest single lever, cheapest to check.
4. **Memory-bound or compute-bound?** (§0) This picks which half of the list is even relevant.
5. **Is occupancy pathologically low?** (§2) Below ~30% investigate; above 50% ignore it.
6. **Is there reuse to exploit?** (§4 tiling) and **contention to break up?** (§5 privatization)
7. **Are threads re-paying a fixed cost?** (§6 coarsening)
8. **Can the algorithm move fewer bytes at all?** (§7) The biggest lever, and the last one, because it's the most work.

The recurring theme, worth stating once plainly: on a modern GPU, arithmetic is close to free and memory traffic is the budget you actually spend. Six of the seven techniques above are about moving fewer bytes, or moving the same bytes in a friendlier order.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — the hardware these techniques exploit
- [INTRINSICS.md](INTRINSICS.md) — the instructions they compile to
- [GLOSSARY.md](GLOSSARY.md) — terminology
- [TASKS.md](TASKS.md) — tasks 94 and 100 are the profiling and %-of-peak exercises for this document
