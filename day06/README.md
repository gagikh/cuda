# Day 6: Streams and Events

## Objectives
- Consolidate the first five days: threads/blocks/grids, memory types, bank conflicts
- Introduce CUDA streams and events
- Use `cudaEvent`s for precise device-side timing (first formal use — earlier days used host-side `<chrono>` on purpose)
- Choose how the host thread waits for the GPU with `cudaSetDeviceFlags`, and know why it must be the first CUDA call
- Enqueue host code into a stream with `cudaLaunchHostFunc`, and explain why it must not call any CUDA API
- Implement image derivative, shared-memory convolution, and transform kernels on a real image loaded via OpenCV

## Key Concepts
- SIMD arch
- Threads/Blocks/Grids
- Global/local/shared/constant memory
- Page locked/pinned memory
- Pitched memory
- DDR and depth: https://depletionmode.com/ram-mapping.html
- Bank conflicts in shared memory
- Streams/events
- Host functions in a stream (`cudaLaunchHostFunc`)
- Host-side wait policy: spin, yield, blocking sync (`cudaSetDeviceFlags`)

## Definitions · Սահմանումներ

*Terms introduced today. Same text as the matching entries in [GLOSSARY.md](../GLOSSARY.md).*

**Stream** — GPU-ի գործողությունների (kernel-ներ, պատճենումներ) կարգավորված հերթ։ Մեկ stream-ի ներսում գործողությունները կատարվում են այն հերթականությամբ, որով ուղարկվել են։ Տարբեր stream-երի գործողությունները կարող են կատարվել միաժամանակ։

**Default stream (per-thread)** — Այն stream-ը, որն օգտագործվում է, երբ stream նշված չէ։ `--default-stream per-thread` flag-ով կոմպիլացնելիս host-ի ամեն thread ստանում է իր default stream-ը, որը չի սպասում մյուս stream-երի գործողությունների ավարտին։

**Event** — Նշիչ, որը `cudaEventRecord`-ով դրվում է stream-ում։ Event-ը համարվում է ավարտված, երբ ավարտվում է այդ stream-ում դրանից առաջ ուղարկված ամբողջ աշխատանքը։ Օգտագործվում է սպասելու (`cudaEventSynchronize`) և ժամանակ չափելու (`cudaEventElapsedTime`) համար։

**Stream-երի կախվածություն** — Տարբեր stream-երի գործողությունների միջև հերթականություն։ Այն սահմանվում է այսպես. մի stream-ում գրանցվում է event, իսկ մյուս stream-ը սպասում է դրան `cudaStreamWaitEvent`-ով։

**Device-ի կողմից ժամանակաչափում** — Ժամանակի չափում event-ներով, ոչ թե host-ի ժամացույցով։ Այդպես չափվում է GPU-ի աշխատանքի ժամանակը, և արդյունքը չի ներառում host-ի կողմից launch-ի ուշացումը։

## Functions · Ֆունկցիաներ

*Interfaces introduced today. Full consolidated list in [API.md](../API.md); concepts in [GLOSSARY.md](../GLOSSARY.md).*

```c
// Ստեղծում է նոր stream
cudaError_t cudaStreamCreate(cudaStream_t *pStream);

// Event-ներ։ cudaEventElapsedTime-ը *ms-ում գրում է start-ի և end-ի միջև ժամանակը միլիվայրկյաններով
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

// stream-ի հետագա գործողությունները սպասում են event-ի ավարտին
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);

// Host-ի ֆունկցիա stream-ում։ Չի կարող կանչել CUDA API
typedef void (CUDART_CB *cudaHostFn_t)(void *userData);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);

// Սպասում է device-ի ամբողջ աշխատանքի ավարտին
cudaError_t cudaDeviceSynchronize(void);

// Որոշում է, թե ինչպես է host-ի թելը սպասում device-ին։ Պետք է կանչել
// մինչև device-ի նախաստորագրումը, այլապես վերադարձնում է cudaErrorSetOnActiveProcess
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int *flags);
```

## Visual
![Single default stream running H2D copy, kernel, D2H copy back to back, versus two streams where one stream's copy overlaps another stream's kernel](streams_timeline.svg)

The default stream runs everything strictly in order — the GPU sits idle during both copies. Once you use two (or more) streams, the copy engine and the compute engine can work at the same time, so one stream's transfer overlaps another stream's kernel. `cudaEvent`s are how you measure exactly how much time that overlap actually saves.

## How the Host Thread Waits: `cudaSetDeviceFlags`

Every `cudaDeviceSynchronize`, `cudaStreamSynchronize` and `cudaEventSynchronize` in this day's lab makes the **host** thread wait for the GPU. How it waits is a choice, and by default the runtime makes it for you:

```c++
cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
```

| Flag | How the host thread waits | Cost |
|---|---|---|
| `cudaDeviceScheduleAuto` | **Default.** Compares active CUDA contexts (C) against logical processors (P): spins if `C ≤ P`, yields if `C > P` | Usually right, but it *is* a heuristic — on a machine with spare cores you silently get spinning |
| `cudaDeviceScheduleSpin` | Busy-waits on the CPU | Notices completion soonest; holds a core at 100% and can slow other host threads |
| `cudaDeviceScheduleYield` | Yields its timeslice back to the OS | Higher latency to notice completion; leaves the core available |
| `cudaDeviceScheduleBlockingSync` | Blocks on a synchronization primitive until the device signals | Lowest CPU use, highest wake-up latency |

Two other flags live in the same call: `cudaDeviceMapHost`, which you need for zero-copy (Day 4), and `cudaDeviceLmemResizeToMax`.

### The ordering trap

The flags apply to a device **before it is initialized**. Call this after the device is already up — after any allocation, launch, or even `cudaSetDevice` in most cases — and you get `cudaErrorSetOnActiveProcess`, with no effect. Recovering from that means `cudaDeviceReset()` first.

So this belongs at the very top of `main`, before anything else touches CUDA. And since it returns a real error that beginners routinely ignore, it is worth checking:

```c++
int main()
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));   // first CUDA call
    ...
}
```

Query what you actually got with `cudaGetDeviceFlags(&flags)` rather than assuming.

### Why this matters on the timing day

The choice does not change how fast the GPU runs. It changes two things that matter here:

- **Measurement noise.** A spinning thread notices completion within nanoseconds; a blocking one can take a scheduler timeslice to wake. For the short kernels you are timing today that wake-up latency is visible, which is one reason `cudaEventElapsedTime` (measured on the device) is more trustworthy than wrapping the launch in a host clock.
- **What else the machine is doing.** Spinning holds a full core. On your own desktop timing one kernel, that is free. On a shared cluster node, or when the host has real work to do in parallel with the GPU — decoding the next video frame, for instance, which is exactly the Day 7 pipeline — spinning steals the core you wanted that work to run on.

Sensible default: leave it on `Auto` and only set it explicitly when you have measured a reason. Reach for `Spin` when latency to notice completion genuinely matters and the host has nothing else to do; reach for `BlockingSync` when the host is busy or the GPU work is long.

### The per-event equivalent

There is a finer-grained version of the same idea. An event created with `cudaEventBlockingSync` makes `cudaEventSynchronize` on *that event* block, regardless of the device-wide flag:

```c++
cudaEventCreateWithFlags(&evt, cudaEventBlockingSync);
```

Useful when most of your waits should spin but one long-running stage should not burn a core. Note that `cudaEventDisableTiming` is a separate flag on the same call — an event created with it cannot be used with `cudaEventElapsedTime`, which is a confusing failure if you set it by habit.

## Host Functions: running CPU code *inside* a stream

Events tell you when GPU work finished, but only if the host asks — `cudaEventSynchronize` blocks, `cudaEventQuery` polls. Sometimes you want the opposite: enqueue a piece of **host** code into the stream so it runs by itself once the preceding work completes, with no host thread waiting around.

```c++
// cudaHostFn_t is: void CUDART_CB fn(void *userData)
void CUDART_CB on_chunk_done(void *userData)
{
    auto *ctx = static_cast<chunk_ctx *>(userData);
    printf("chunk %d ready\n", ctx->index);   // host work only
}

cudaMemcpyAsync(d_in, h_in, n, cudaMemcpyHostToDevice, stream);
process<<<grid, block, 0, stream>>>(d_in, d_out, n);
cudaMemcpyAsync(h_out, d_out, n, cudaMemcpyDeviceToHost, stream);

CUDA_CHECK(cudaLaunchHostFunc(stream, on_chunk_done, &ctx));   // runs after the copy-out
```

The call returns immediately. The function itself runs later, on a driver-internal thread, after everything already queued in that stream has completed.

### The rule that makes this dangerous

**The host function must not make any CUDA API call.** Not `cudaMalloc`, not a kernel launch, not `cudaMemcpyAsync`, not `cudaStreamSynchronize` — nothing. The documentation is explicit that attempting one *may* return `cudaErrorNotPermitted`, "but this is not required." In other words the failure mode is undefined behaviour or a deadlock, not a tidy error code you can check.

That restriction is less arbitrary than it looks. Your function is running *as* a stream operation, on a driver thread, in the middle of the driver's own bookkeeping. Calling back into the driver from there is re-entrancy, and asking it to wait on GPU work would be asking the stream to wait on itself.

Three more properties worth knowing before you use one:

- **It blocks the rest of the stream.** Work enqueued after the host function does not start until the function returns. A slow callback stalls the GPU, which is the opposite of what you came to Day 6 for. Keep it to a flag, a counter, a `printf`, or a push onto a lock-free queue.
- **Order across streams is undefined.** Host functions in independent streams may run in any order, and may be serialized against each other.
- **It does not run if the context has errored.** So it is a completion notification, not an error handler — report failures through `CUDA_CHECK` on the host side as usual.

### Why this belongs on the streams day

A host function is **capturable**: during `cudaStreamBeginCapture` / `cudaStreamEndCapture` it becomes a host node in the graph. That means a whole pipeline — copy in, kernel, copy out, notify the host — can be recorded once and replayed as a single CUDA graph on Day 12, with the notification still firing at the right point in the sequence. Without it, the "tell me when this chunk is done" step would have to live outside the graph, and the pipeline could never be captured as one unit.

## Resources
https://www.cse.iitd.ac.in/~rijurekha/col730_2022/cudastreams_aug25_aug29.pdf
https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf

## Hands-On Task
- Implement image derivatives, on a real image loaded via `cv::imread` and uploaded to `cv::cuda::GpuMat` (see [`template.cu`](template.cu))
- Implement convolution via shared memory
- Implement image transform

## Self-Learning
1. Implement an image derivative (gradient) kernel — compute dx/dy per pixel. `dx`/`dy` are float `GpuMat`s in the template; `cv::normalize` (or take the absolute value and scale) before `cv::imshow`, or the result will look black.
2. Implement convolution via shared memory (reuse your Day 5 tiling approach).
3. Implement a simple image transform (e.g. rotate or scale) kernel.
4. Time each kernel precisely with `cudaEvent`s (`cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime`) and compare against your earlier `<chrono>` measurements.
5. (Stretch) Split the derivative + transform work across two CUDA streams and check whether they overlap.
5b. Time the same kernel under `cudaDeviceScheduleSpin` and `cudaDeviceScheduleBlockingSync`, watching host CPU usage (`top` / Task Manager) in each case. Compare the `cudaEvent` time against a `<chrono>` time around the same launch, and say which of the two numbers the flag actually moved.
5c. Call `cudaSetDeviceFlags` *after* your first allocation instead of before it. Confirm you get `cudaErrorSetOnActiveProcess`, and that without `CUDA_CHECK` you would never have noticed.
6. Enqueue a `cudaLaunchHostFunc` after the copy-out that prints a message, and confirm from the ordering of your `printf`s that it runs *after* the GPU work rather than at the point you called it.
7. Put a `sleep` of a few hundred milliseconds inside that host function and time the stream again. Explain the slowdown in terms of "the host function blocks work added after it".
8. Call `cudaMalloc` (or any CUDA API) from inside the host function. Does it return `cudaErrorNotPermitted`, hang, or appear to work? Note what you observed — the documentation permits all three.

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. Why does the default stream serialize operations even when they don't depend on each other?
2. What would go wrong if you called `cudaEventElapsedTime()` right after `cudaEventRecord(stop)`, without `cudaEventSynchronize(stop)` first?
3. Why do the `dx`/`dy` gradient outputs need `cv::normalize` (or similar) before `cv::imshow`, when the filtered image from Day 5 didn't?
4. `cudaLaunchHostFunc` returns immediately, but the function it enqueues runs much later. What decides *when*?
5. Why is a host function forbidden from calling into the CUDA API at all? Think about which thread it runs on and what it would be asking the driver to do.
6. A host function does not run if the context has errored. What does that tell you about where error handling belongs in a streamed pipeline?
7. `cudaSetDeviceFlags` changes nothing about how fast the GPU executes. What does it change, and why is that still worth caring about on Day 7's video pipeline?
8. On a machine with 16 cores running one CUDA context, which behaviour does `cudaDeviceScheduleAuto` pick — and is that what you want on a shared cluster node?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.

No CUDA GPU on your machine? Run this lab in the [course Colab notebook](https://colab.research.google.com/drive/1zDtYkz8WwD7sOIucSyUoxm2n7RYWJxVZ?usp=sharing) instead — free T4, compute capability 7.5, which is exactly the `-arch=sm_75` the template compiles for. One caveat for this day: Colab's preinstalled OpenCV is CPU-only, so `cv::cuda::GpuMat` will not link. Either build OpenCV with `-DWITH_CUDA=ON` in the notebook, or swap the image I/O for a `cudaMalloc` buffer and a synthetic image — the CUDA content of this day is unchanged either way. Full setup in the [root README](../README.md#-no-cuda-gpu-start-here).
