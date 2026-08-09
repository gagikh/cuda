# CUDA Intrinsics Cheat Sheet

Device-side intrinsics this course uses, plus the near neighbours you'll reach for straight after. Each entry notes the day it's introduced (`(Day N)`), or `(beyond)` if the course doesn't cover it but you'll meet it the moment you read real CUDA code.

An **intrinsic** is a function the compiler turns into one specific machine instruction (or a short, known sequence) instead of a real function call. That's the whole appeal: `__popc(x)` isn't a bit-counting loop, it's a single `POPC` instruction. See [ARCHITECTURE.md](ARCHITECTURE.md#from-cc-to-sass-instruction-reference) for what each one compiles to.

**One rule before the tables:** every warp-level intrinsic below takes a `mask` of participating lanes, and every lane in that mask must reach the call. Pass `0xFFFFFFFF` only when you actually know all 32 lanes are live — inside a divergent branch or a bounds check, that's a lie, and the result is undefined behaviour on Volta and newer (independent thread scheduling means lanes genuinely can be at different instructions). Use `__activemask()` to ask which lanes are really here, or restructure so the intrinsic sits outside the branch.

---

## Warp shuffle — move a register between lanes, no memory at all

| Intrinsic | What it does |
|---|---|
| `__shfl_sync(mask, var, srcLane)` | Every lane reads `var` from lane `srcLane`. The general form; the three below are specialisations. *(Day 8)* |
| `__shfl_down_sync(mask, var, delta)` | Lane `i` reads from lane `i + delta`. The reduction workhorse: halve `delta` from 16 down to 1 and lane 0 ends up with the warp's total in 5 steps. *(Day 8)* |
| `__shfl_up_sync(mask, var, delta)` | Lane `i` reads from lane `i - delta`. The scan (prefix-sum) direction. *(Day 8)* |
| `__shfl_xor_sync(mask, var, laneMask)` | Lane `i` swaps with lane `i ^ laneMask`. Butterfly exchange — every lane both sends and receives, so all 32 lanes end up with the result instead of just lane 0. *(Day 8)* |

Lanes reading from an inactive or out-of-range lane get their own value back, not garbage — which is why the classic `for (offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(...)` is correct without any bounds checking.

## Warp vote & mask — ask a question of all 32 lanes at once

| Intrinsic | What it does |
|---|---|
| `__ballot_sync(mask, pred)` | Returns a 32-bit word, bit `i` set if lane `i`'s `pred` was true. Packing 32 booleans into one integer in one instruction. *(Day 9)* |
| `__all_sync(mask, pred)` | Non-zero if `pred` was true on *every* participating lane. *(Day 9)* |
| `__any_sync(mask, pred)` | Non-zero if `pred` was true on *any* participating lane. *(Day 9)* |
| `__activemask()` | Which lanes are actually executing right now. Read it, don't guess `0xFFFFFFFF`. *(beyond)* |
| `__match_any_sync(mask, value)` | Groups lanes by value: each lane gets a mask of the other lanes holding the same `value`. Volta+. The efficient way to do per-key aggregation inside a warp before hitting `atomicAdd`. *(beyond)* |

`__ballot_sync` + `__popc` is the standard "how many lanes passed?" pair, and `__ballot_sync` + `__popc(ballot & lanemask_lt)` gives each lane its exclusive prefix — a whole warp scan in two instructions, which is what Day 8's index-compaction task is really after.

## Bit manipulation

| Intrinsic | What it does |
|---|---|
| `__popc(x)` / `__popcll(x)` | Population count: number of set bits, 32- and 64-bit. `__popc(a ^ b)` *is* Hamming distance. *(Day 10)* |
| `__clz(x)` / `__clzll(x)` | Count leading zeros. `31 - __clz(x)` is floor(log2(x)) for `x > 0`. *(beyond)* |
| `__ffs(x)` | Find first set: 1-based index of the lowest set bit, 0 if `x == 0`. Iterating a `__ballot_sync` result one lane at a time. *(beyond)* |
| `__brev(x)` | Reverse the bit order. Turns up in FFT and radix-sort index math. *(beyond)* |
| `__byte_perm(a, b, s)` | Arbitrary byte-level shuffle across two words. *(beyond)* |
| `__funnelshift_l/r(lo, hi, s)` | Shift a 64-bit value made of two 32-bit halves, keeping 32 bits. Cheap misaligned loads from registers. *(beyond)* |

## Math — explicit rounding, and fast approximations

The `_r*` suffix picks the IEEE rounding mode: `_rn` nearest-even, `_rz` toward zero, `_ru` up, `_rd` down.

| Intrinsic | What it does |
|---|---|
| `__fmaf_rn(a, b, c)` | Fused multiply-add, one rounding. The compiler already emits `FFMA` for `a*b + c` — write this only to pin the rounding down, or to keep the fusion when compiling with `-fmad=false`. |
| `__fadd_rn`, `__fmul_rn`, `__fsub_rn` | Basic ops with the rounding mode nailed down, which also blocks the compiler from fusing them into an FMA. Reach for these when reproducibility matters more than speed. |
| `__fdividef(a, b)` | Fast approximate division via the SFU. Much cheaper than `a / b`; loses accuracy for extreme operands. |
| `__expf`, `__logf`, `__sinf`, `__cosf`, `__powf` | SFU-routed approximate transcendentals. `--use_fast_math` silently swaps your plain `expf`/`logf`/... calls for exactly these — using the `__`-prefixed names instead opts individual call sites in, which is usually the better trade. |
| `rsqrtf(x)` | Reciprocal square root in roughly one instruction. Meaningfully cheaper than `1.0f / sqrtf(x)`. |
| `__saturatef(x)` | Clamp to [0, 1] for free. Handy in image kernels (Day 11, Day 15) where you're normalising anyway. |
| `min`, `max`, `abs`, `fminf`, `fmaxf`, `fabsf` | Single instructions, not branches. Never hand-roll these with an `if`. |
| `__half2float`, `__float2half`, `__hadd`, `__hmul` | FP16 conversion and arithmetic (`cuda_fp16.h`). The entry point to task 98 in [TASKS.md](TASKS.md). *(beyond)* |

Rule of thumb: write plain arithmetic first and read the SASS (`cuobjdump --dump-sass`) before reaching for any of these. The compiler is already generating `FFMA`, `MUFU.RSQ` and friends on its own; fighting it for instructions it would emit anyway is wasted effort.

## Memory & cache

| Intrinsic | What it does |
|---|---|
| `__ldg(ptr)` | Load through the read-only / texture data cache instead of the normal L1 path. Good for data that's read-only for the whole kernel and read by many threads. On Kepler+ the compiler often does this for you when a pointer is marked `const __restrict__`. *(Day 13)* |
| `__ldca`, `__ldcg`, `__ldcs`, `__ldlu`, `__ldcv` | Loads with an explicit cache-eviction hint: default / L2-only / evict-first / last-use / volatile. Full table with "when to reach for it" in [ARCHITECTURE.md](ARCHITECTURE.md#cache-eviction-hints-lru-and-loadstore-cache-operators). *(Day 13)* |
| `__stwb`, `__stcg`, `__stcs`, `__stwt` | The store-side counterparts: write-back / L2-only / evict-first / write-through. *(Day 13)* |
| `__prefetch_global_l2(ptr)` | Ask for a line to be pulled into L2 ahead of use. *(beyond)* |
| `tex2D<T>(texObj, x, y)` | Sample a texture object — bilinear filtering and address clamping happen in the texture unit, in hardware, on the way out. *(Day 11)* |
| `surf2Dread` / `surf2Dwrite` | Read/write a surface object. Unlike textures, surfaces are writable. *(Day 11)* |

## Atomics

All return the value that was in memory *before* the operation.

| Intrinsic | What it does |
|---|---|
| `atomicAdd(ptr, val)` | The one you'll use most. Supports `int`, `unsigned`, `unsigned long long`, `float`, and `double` (compute capability ≥ 6.0). *(Day 9)* |
| `atomicSub`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExch` | Same shape, different operation. *(Day 9)* |
| `atomicCAS(ptr, compare, val)` | Compare-and-swap. The primitive you build every other atomic out of when there's no built-in for your type — the standard trick for an atomic float min/max. *(beyond)* |
| `atomicAdd_block(ptr, val)` | Block-scoped atomic — cheaper, because it only has to be coherent within the block rather than device-wide. *(beyond)* |

Atomics resolve at L2, not at the issuing SM (see [ARCHITECTURE.md](ARCHITECTURE.md)), so heavy contention on one address serializes chip-wide regardless of how many SMs are involved. The standard mitigation is the Day 8/9 pattern: warp-reduce first, then have one lane per warp do the `atomicAdd` — 32× fewer atomic requests for the same answer.

## Synchronization

| Intrinsic | What it does |
|---|---|
| `__syncthreads()` | Block-wide barrier **plus** a memory fence on shared and global memory. Every thread in the block must reach it — calling it inside a divergent branch is undefined behaviour, not merely slow. *(Day 5)* |
| `__syncwarp(mask)` | Warp-level barrier. Needed on Volta and newer, where lanes in a warp can genuinely be at different instructions, whenever you hand-write lane-to-lane communication through shared memory rather than shuffles. *(Day 9)* |
| `__syncthreads_count(pred)` | Barrier that also returns how many threads in the block had `pred` true. *(beyond)* |
| `__syncthreads_and(pred)` / `__syncthreads_or(pred)` | Barrier plus a block-wide AND / OR of the predicate. *(beyond)* |
| `__threadfence()` / `__threadfence_block()` / `__threadfence_system()` | Memory-ordering fences without a barrier — device-wide, block-wide, and system-wide (including the host). Ordering only; they don't make anyone wait for anyone else. *(beyond)* |

The distinction worth being precise about: `__syncthreads()` guarantees that every thread in the *block* has arrived and that their shared/global writes are visible to the block. It says nothing about other blocks, and nothing about whether the data is visible to the host.

## Thread & lane identity

| Expression | What it gives you |
|---|---|
| `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | The built-in `dim3`s. `blockIdx.x * blockDim.x + threadIdx.x` is the global index formula from Day 2. *(Day 2)* |
| `warpSize` | Built-in variable, always 32 today. Use it instead of hardcoding 32 — the cost is zero and the intent is clearer. *(Day 3)* |
| lane id | `threadIdx.x & 31` **only** when `blockDim.x` is a multiple of 32. In general, flatten first: `(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) & 31`. Getting this wrong is the most common bug in hand-written warp code. *(Day 8)* |
| `%laneid`, `%lanemask_lt` | PTX special registers, reachable via inline `asm` or `cooperative_groups`. `__popc(ballot & lanemask_lt)` is the two-instruction warp exclusive scan. *(beyond)* |
| `printf(...)` | Yes, device-side `printf` works, and it's the fastest way to debug a kernel. Output is buffered and flushed at the next synchronization. *(Day 1)* |
| `assert(...)` | Also works on device; a failed assert kills the kernel and puts the context into an error state you'll see on the next `CUDA_CHECK`. *(beyond)* |

## Cooperative Groups — the modern wrapper *(beyond this course)*

`#include <cooperative_groups.h>` gives typed group objects instead of raw masks, which makes the "which lanes are participating?" question structural rather than a comment you hope stays true:

```c++
namespace cg = cooperative_groups;

__device__ int warp_reduce(int val)
{
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);   // mask handled for you
    }
    return val;                                // every lane holds the sum
}
```

`cg::this_grid().sync()` additionally gives a grid-wide barrier without a host round-trip, if the kernel is launched with `cudaLaunchCooperativeKernel`. Tasks 91 and 97 in [TASKS.md](TASKS.md) are the on-ramp.

---

## Where these show up in the course

| Day | Intrinsics introduced |
|---|---|
| Day 1 | device `printf` |
| Day 2 | `threadIdx` / `blockIdx` / `blockDim` / `gridDim` |
| Day 3 | `warpSize`, `#pragma unroll` |
| Day 5 | `__syncthreads()` |
| Day 8 | `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` |
| Day 9 | `__ballot_sync`, `__all_sync`, `__any_sync`, `atomicAdd`, `__syncwarp` |
| Day 10 | `__popc` |
| Day 11 | `tex2D`, `surf2Dread` / `surf2Dwrite` |
| Day 13 | `__ldg`, `__ldcs` / `__stcs` and the rest of the cache-operator family |
| Day 14 | `curand_uniform`, `curand_init` (library, not strictly intrinsics) |

See also: [GLOSSARY.md](GLOSSARY.md) for the concepts, [ARCHITECTURE.md](ARCHITECTURE.md) for the hardware these instructions run on, and [TASKS.md](TASKS.md) for exercises that use them.
