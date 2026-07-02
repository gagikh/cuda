# Day 13: Cache Behavior and Optimization

## Objectives
- Deepen understanding of shared-memory bank conflicts and how to remove them
- Understand L1/L2 cache behavior and persistence hints
- Use `__ldg` to hint read-only global memory access
- Understand LRU cache eviction and how to bias it with per-instruction cache operators
- Apply the week's techniques to optimize a real kernel

## Key Concepts
- Bank conflicts
- Using L2 cache
- Persistent cache for compiled programs and configuration
- `__ldg` forces the compiler to consider memory read-only
- LRU cache eviction, and per-instruction cache hints (`__ldcs`, `__stcs`, `__ldlu`, ...)

## Visual
![Memory hierarchy pyramid: registers (fastest, smallest) at top, then shared memory / L1, then L2, then global memory / VRAM (slowest, largest) at bottom](cache_hierarchy.svg)

Every optimization this day is about the same idea: keep frequently-read data as high in this pyramid as possible, for as long as possible. `__ldg()`, shared-memory swizzling, and L2 persistence hints are three different tools for the same goal.

![XOR swizzling: an 8x8 shared-memory tile where accessing logical column 3 without swizzling always hits bank 3 for every row, but indexing with tile[row][col ^ row] spreads that same logical column across a different bank on every row, with no padding column needed](swizzling.svg)

Padding (Day 5) fixes bank conflicts by wasting a column so the row stride is no longer a multiple of the bank count. Swizzling fixes the same problem without wasting any memory: index shared memory as `tile[row][col ^ row]` instead of `tile[row][col]`. Because XOR is its own inverse, writing and reading with the same swizzle formula is still correct — you just physically scatter each logical column across every bank instead of pinning it to one.

For the hardware behind this pyramid — why shared memory and L1 are the same physical SRAM, why L2 is shared across every SM instead of being per-SM like L1, and where atomic operations actually get resolved — see [ARCHITECTURE.md](../ARCHITECTURE.md).

## LRU and Per-Instruction Cache Hints
L1 and L2 are both finite, so when full, something gets evicted to make room for a new line. The hardware's replacement policy is an approximation of **LRU (Least Recently Used)**: evict whichever line hasn't been touched in the longest time. `cudaAccessPolicyWindow` (below) biases that policy for a whole buffer at once; there's also a finer-grained tool — per-instruction cache operators — that biases it one load or store at a time:

```c++
float x = __ldcs(&input[i]);   // "streaming" read: touched once, evict-first hint
__stcs(&output[i], result);    // "streaming" write: won't be re-read, don't linger in cache
```

`__ldcs`/`__stcs` are the two you'll reach for most: mark data you're touching exactly once so it doesn't crowd out data your kernel (or a neighboring warp) genuinely reuses. This is a *different* knob from `__ldg()` — `__ldg` changes *which cache* a read uses (the read-only/texture path, Day 11); `__ldcs`/`__stcs` stay on the normal L1/L2 path and change *how eagerly the replacement policy discards the line*. Full intrinsic list (`__ldca`, `__ldcg`, `__ldcs`, `__ldlu`, `__ldcv`, `__stwb`, `__stcg`, `__stcs`, `__stwt`) and PTX mnemonics in [ARCHITECTURE.md](../ARCHITECTURE.md#cache-eviction-hints-lru-and-loadstore-cache-operators).

## Resources
https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

https://cuda-programming.blogspot.com/2013/02/bank-conflicts-in-shared-memory-in-cuda.html

Shared memory swizzling reference:
https://leimao.github.io/images/blog/2024-05-14-CUDA-Shared-Memory-Swizzling/swizzling.png

## Hands-On Task
Optimize transform (the Day 6 image transform kernel), on a real image loaded via `cv::imread` / `cv::cuda::GpuMat`.

## Self-Learning
1. Add `__ldg()` to a read-heavy kernel from an earlier day (e.g. the Day 5 tiled filter) and measure the effect.
2. Fill in `tiled_filter_swizzled` in [`template.cu`](template.cu): use `tile[row][col ^ row]` (both `[TILE_DIM][TILE_DIM]`, no padding column) for every shared-memory read and write, and compare its timing against the padded Day 5 version.
3. Experiment with L2 persistence hints (`cudaAccessPolicyWindow`) on a buffer that's read repeatedly across kernel launches.
4. Optimize the Day 6 image transform kernel using everything from this week (shared memory, texture, `__ldg`, bank-conflict-free layout) and document before/after timings.
5. In one of your filter kernels, write the final output with `__stcs` instead of a plain store (it's written once and never read back inside the kernel), and read the tile's halo region with the default `__ldca` (it's reused by neighboring threads). Measure whether marking the write-once output as streaming changes anything.

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. Why does `__ldg` only help for data the kernel treats as read-only?
2. Why does `col ^ row` swizzling need the row width to be a power of two to cleanly avoid bank conflicts?
3. What's the tradeoff L2 persistence hints (`cudaAccessPolicyWindow`) are making — and when could they make performance *worse* instead of better?
4. Why is the hardware's LRU replacement policy an *approximation* rather than exact LRU, and why does that make cache hints like `__ldcs` useful even when the hardware is "supposed" to figure out reuse patterns on its own?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
