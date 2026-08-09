# Day 5: Memory Conflicts and Shared Memory

## Objectives
- Understand shared memory banking and how bank conflicts happen
- Correctly synchronize reads/writes to shared memory (`__syncthreads()`)
- Distinguish global, shared, constant, and pitched memory and when to use each
- Implement a shared-memory tiled filter
- Load/display real images and video with OpenCV, and process them via `cv::cuda::GpuMat`

## Key Concepts
- Bank conflicts
- Sync read/write in kernel
- Global memory
- Shared memory
- Constant memory
- Pitched memory
- `cv::cuda::GpuMat`, `cv::imread`, `cv::VideoCapture`, `cv::imshow`

## Visual
![Conflict-free shared memory access where each thread hits a different bank, versus a bank conflict where multiple threads hit bank 0 due to stride-32 access](bank_conflicts.svg)

Shared memory is split into 32 banks so that 32 threads can be serviced in one transaction — but only if each thread hits a different bank. Stride-32 access patterns (common when indexing by a tile width that's a multiple of 32) collapse onto the same bank and get serialized. Padding the row stride by one element is the standard fix, and it's exactly what `tiled_filter` in [`template.cu`](template.cu) is set up for.

## Animated
![A 32-bank histogram cycling through four access strides: stride 1 gives one thread per bank, stride 2 piles two threads on every even bank, stride 4 piles four threads on every fourth bank, and stride 32 piles all 32 threads onto bank 0](bank_conflict_nway.svg)

Four strides, same warp, same 32 banks. The cost of a shared-memory access is the **tallest bar**, not the number of bars: whatever the worst-hit bank has to serve, every other lane waits for. Stride 1 → 1 transaction, stride 2 → 2, stride 4 → 4, stride 32 → 32.

The general rule behind the picture: lane `t` lands on bank `(t × stride) mod 32`, so the conflict degree is exactly **`gcd(stride, 32)`**. That has a consequence worth internalizing — *every odd stride is conflict-free*. Stride 3, 7, 17 and 31 all cost one transaction, identical to stride 1. Only strides sharing a factor of two with 32 hurt, and each extra factor doubles the damage. Padding a tile to `[TILE][TILE+1]` is nothing more cunning than forcing an even row stride to become odd.

For a configurable version — drag the stride to any value 1–33, sweep it, and step through the transpose read/write phases under all three layouts (plain, padded, swizzled) — see [`bank_conflict_animations.html`](bank_conflict_animations.html), open locally in a browser since GitHub's file viewer only shows HTML as source rather than running it. The transpose tab is the concrete setup for Day 12's `transpose_shared` kernel.

## OpenCV Basics
Starting today, day templates load real images/video through OpenCV instead of filling synthetic buffers by hand. Four things to know:

- **`cv::imread(path, flags)`** — loads an image file into a host-side `cv::Mat`. `cv::IMREAD_GRAYSCALE` gives you a single-channel `unsigned char` image, the simplest thing to feed a kernel.
- **`cv::VideoCapture`** — `cv::VideoCapture cap(path_or_device_index); cv::Mat frame; cap >> frame;` pulls one frame at a time from a video file or camera, in a loop. This is what Day 5's "video stream" final task and Day 6/13's transform tasks are built around.
- **`cv::imshow("window name", mat)` + `cv::waitKey(ms)`** — displays a `cv::Mat` in a window. `waitKey` isn't optional decoration: it pumps the GUI event loop, so nothing actually paints on screen without it. `waitKey(0)` waits for a keypress; `waitKey(1)` is what you want inside a video loop so playback doesn't stall.
- **`cv::cuda::GpuMat`** — the device-side counterpart of `cv::Mat`. `gpuMat.upload(hostMat)` / `gpuMat.download(hostMat)` copy data across the host/device link (Day 1's PCIe/NVLink picture). Critically, a `GpuMat`'s rows are **pitched**, exactly like `cudaMallocPitch` from [`examples/matrix_add.cu`](../examples/matrix_add.cu): `gpuMat.step` is the row stride in bytes, and it's normally larger than `cols * elemSize()` for alignment. Every kernel that touches a `GpuMat` directly has to index rows by `step`, not by `width` — get this wrong and you'll read garbage past the end of narrow images.

Build note: you'll need OpenCV built with its CUDA module (`opencv_cudaarithm`, `opencv_cudaimgproc`, `opencv_highgui`, `opencv_videoio`). With pkg-config: `` `pkg-config --cflags --libs opencv4` ``.

## Resources
http://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec35.html

Task reference: https://developer.download.nvidia.com/compute/DevZone/C/html_x64/3_Imaging/convolutionSeparable/doc/convolutionSeparable.pdf

## Reference Implementation
[`examples/matrix_add.cu`](../examples/matrix_add.cu) at the repo root uses `cudaMallocPitch` / `cudaMemcpy2D` — a working example of pitched memory referenced in this day's material, and the same pitch idea `GpuMat::step` is built on.

## Hands-On Task
Use shared memory for a 2D filter, loaded from a real image via `cv::imread`. Final task: 2D Sobel filter implementation on a video stream via `cv::VideoCapture`.

## Self-Learning
1. Implement a shared-memory tile-based 2D convolution filter (start with a simple box blur) operating on a `cv::cuda::GpuMat` loaded from a real image.
2. Deliberately create a shared-memory access pattern with bank conflicts, measure the perf hit, then fix it with padding.
3. Implement a 2D Sobel filter using shared memory.
4. Extend the Sobel filter to process a video stream frame by frame using `cv::VideoCapture`, displaying the result with `cv::imshow` each frame.

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. Why do 32 threads reading `tile[threadIdx.x][k]` for a fixed `k` all collide on the same shared-memory bank?
1b. A warp accesses shared memory with stride 3. How many transactions does it cost, and why isn't the answer 3? (Then check yourself against the stride explorer in [`bank_conflict_animations.html`](bank_conflict_animations.html).)
2. Why does `GpuMat::step` differ from `cols * elemSize()`, and what breaks in a kernel that ignores that and assumes rows are contiguous?
3. What does `__syncthreads()` actually guarantee, and what does it explicitly *not* guarantee?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.
