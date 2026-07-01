# Day 9: Warp-Level Data Exchange

## Objectives
- Use `__syncwarp`, `__activemask`, and `__ballot_sync` correctly
- Combine warp reduction with atomic operations
- Implement warp-level bit packing/unpacking

## Key Concepts
Warp level programming and `__syncwarp`, `__activemask`, `__ballot_sync`.

## Visual
![__ballot_sync collecting each of the warp's 32 boolean predicates into a single 32-bit mask, one bit per lane](warp_ballot.svg)

`__ballot_sync` turns "which lanes satisfy this condition?" into a single 32-bit integer that every lane in the warp receives — bit N set iff lane N's predicate was true. That mask is exactly what you need for this day's zip/unzip task, and it's the building block `__activemask`/`__syncwarp` use internally to know which lanes are still participating.

## Resources
https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

## Hands-On Task
- Calculate the mean of a real image, loaded via `cv::imread` into a `cv::cuda::GpuMat` (use warp reduction and `atomicAdd`, maybe exchange?)
- `pyrUp`/`pyrDown` functions (https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html)
- Zip/Unzip binary images by 32x (warp level)

## Self-Learning
1. Compute an image's mean pixel value using warp-level reduction, then `atomicAdd` the per-warp partial sums into a single global accumulator. [`template.cu`](template.cu) loads a real image into a `GpuMat` for this.
2. Use `__ballot_sync` to pack 32 binary pixel values into one 32-bit word, and write the inverse (unzip) operation.
3. Implement `pyrDown` (blur + downsample by 2) on the loaded `GpuMat`, and display the result with `cv::imshow`.
4. Implement `pyrUp` (upsample by 2 + blur).

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. What does the 32-bit mask returned by `__ballot_sync` actually represent, bit by bit?
2. Why does computing an image mean use `atomicAdd` into one global accumulator instead of having each warp write into its own slot of a shared array?
3. Why does `pyrDown` blur before downsampling instead of just keeping every other pixel directly?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
