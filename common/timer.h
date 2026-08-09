// common/timer.h
//
// cudaEvent-based kernel timer with the derived performance metrics the
// course actually asks for: achieved bandwidth, achieved TFLOP/s, and each
// as a percentage of this GPU's theoretical peak.
//
//   #include "../common/timer.h"
//
// Why events and not <chrono>: kernel launches are asynchronous, so a host
// clock measures how long the *launch* took to enqueue, not how long the
// kernel ran. cudaEvents are recorded in the stream itself and timed by the
// GPU (Day 6). <chrono> is fine for coarse host-side comparisons (Day 1,
// Day 3); everything past Day 6 should use this.
//
// Typical use:
//
//   kernel_timer_t t;
//   for (int i = 0; i < 100; ++i) {          // warm up + average: a single
//       t.start();                           // measurement is mostly noise
//       my_kernel<<<grid, block>>>(...);
//       t.stop();
//   }
//   t.report("my_kernel");
//   printf("%.2f TFLOP/s\n", t.tflops(2.0 * M * M * M));
//
#pragma once

#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>
#include "cuda_check.h"

struct kernel_timer_t
{
    kernel_timer_t()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~kernel_timer_t()
    {
        // Destructors must not exit() on failure, so these aren't CUDA_CHECK'd
        // (same reasoning as the RAII wrappers in Day 11 / Day 12).
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    kernel_timer_t(const kernel_timer_t&) = delete;
    kernel_timer_t& operator=(const kernel_timer_t&) = delete;

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Blocks until the work between start() and here has finished, then folds
    // the elapsed time into the running statistics.
    void stop(cudaStream_t stream = 0)
    {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));

        total_ms_ += static_cast<double>(ms);
        if (ms < min_ms_) min_ms_ = ms;
        if (ms > max_ms_) max_ms_ = ms;
        ++count_;
    }

    void reset() { total_ms_ = 0.0; min_ms_ = FLT_MAX; max_ms_ = 0.0; count_ = 0; }

    int    count()  const { return count_; }
    // NOTE the double: doing this division in an integer type truncates, and a
    // sub-millisecond kernel then reports 0 ms and an infinite TFLOP/s.
    double get_avg() const { return count_ ? total_ms_ / count_ : 0.0; }
    double get_min() const { return count_ ? min_ms_ : 0.0; }
    double get_max() const { return count_ ? max_ms_ : 0.0; }

    // ---- derived metrics -------------------------------------------------

    // Achieved bandwidth in GB/s. `bytes` is everything the kernel reads PLUS
    // everything it writes, per launch. For c[i] = a[i] + b[i] over n floats
    // that's 3 * n * sizeof(float).
    double gb_per_s(double bytes) const
    {
        const double ms = get_avg();
        return ms > 0.0 ? bytes / (ms * 1e6) : 0.0;   // bytes/ms -> GB/s
    }

    // Achieved TFLOP/s. `flops` is the useful operation count per launch.
    // For an M x M x M matmul: 2.0 * M * M * M  (multiply AND add -- the
    // factor of 2 is the convention cuBLAS is benchmarked with; drop it and
    // you'll report half your real throughput).
    //
    // Pass this as a double. `2 * M * M * M` computed in int overflows at
    // M = 1291; write `2.0 * M * M * M` so the promotion happens first.
    double tflops(double flops) const
    {
        const double ms = get_avg();
        return ms > 0.0 ? flops * 1e-9 / ms : 0.0;    // flops/ms -> TFLOP/s
    }

    // Convenience for the common square-matmul case.
    double tflops_matmul(int M) const { return tflops(2.0 * M * M * M); }

    // ---- theoretical peaks, for the ratio that actually guides you -------

    // Peak global-memory bandwidth in GB/s (same formula device_info.h uses).
    //
    // Every term earns its place -- work through it once and you'll never
    // have to look it up again:
    //
    //   p.memoryClockRate       UNIT: kilohertz. Not Hz, not MHz -- the CUDA
    //                           runtime reports clocks in kHz. An RTX 4090
    //                           reports 10501000.
    //
    //   / 1000.0                kHz -> MHz, i.e. millions of cycles/second.
    //                           10501000 kHz -> 10501 MHz.
    //
    //   p.memoryBusWidth        UNIT: BITS. How many bits the memory bus
    //                           moves per transfer. 384 on a 4090, 5120 on
    //                           an A100 (HBM is very wide and comparatively
    //                           slow; GDDR is narrow and fast).
    //
    //   / 8                     bits -> bytes, because bandwidth is quoted
    //                           in bytes. 384 bits -> 48 bytes per transfer.
    //                           Integer division, which is safe: every real
    //                           bus width (64/128/192/256/384/512/5120) is a
    //                           multiple of 8.
    //
    //   2.0 *                   DDR -- Double Data Rate. GDDR and HBM both
    //                           transfer on BOTH the rising and the falling
    //                           edge of each clock cycle, so one cycle moves
    //                           two bus-widths of data, not one. This factor
    //                           is about the memory technology, and is NOT
    //                           the same "2" as in the peak-FLOPs formula
    //                           below (that one is the FMA counting as two
    //                           operations). Unrelated coincidence.
    //
    //   / 1000.0                MHz * bytes already gives MB/s (10^6
    //                           transfers/s * bytes each), so one more
    //                           factor of 1000 lands on GB/s.
    //
    // Worked, for a 4090:
    //   10501000 kHz / 1000        = 10501 MHz
    //   384 bits / 8               = 48 bytes
    //   2 * 10501 * 48             = 1,008,096 MB/s
    //   / 1000                     = 1008 GB/s   <- matches the spec sheet
    //
    // Verified the same way against A100 (1935), 3090 (936), V100 (900) and
    // H100 SXM (3350). Note GB here means 10^9 bytes, not 2^30 -- the decimal
    // convention every vendor quotes bandwidth in.
    //
    // This is a THEORETICAL ceiling: the number the bus could sustain if it
    // never idled. Real kernels top out around 80-90% of it even when
    // perfectly coalesced, because of refresh, row activation, and read/write
    // turnaround. Treat >80% as "done", not as "20% left on the table".
    static double peak_gb_per_s(int device = 0)
    {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, device));
        return 2.0 * (p.memoryClockRate / 1000.0) * (p.memoryBusWidth / 8) / 1000.0;
    }

    // Peak FP32 in TFLOP/s. FP32 lanes per SM is NOT queryable through
    // cudaDeviceProp -- it's 64 on compute 7.0/8.0 (V100, A100) and 128 on
    // 7.5/8.6/8.9 and Ada/Blackwell consumer parts. Same caveat as the
    // tensor-core count in device_info.h: an educated guess, not a fact.
    static double peak_tflops_fp32(int device = 0)
    {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, device));

        int lanes = 128;
        if (p.major == 7 && p.minor == 0) lanes = 64;   // Volta
        else if (p.major == 8 && p.minor == 0) lanes = 64; // A100
        else if (p.major == 6 && p.minor == 0) lanes = 64; // P100

        const double ghz = p.clockRate / 1e6;           // kHz -> GHz
        return 2.0 * p.multiProcessorCount * lanes * ghz / 1000.0;
    }

    // ---- reporting -------------------------------------------------------

    void report(const char *name) const
    {
        printf("%-24s avg %8.3f ms  min %8.3f  max %8.3f  (n=%d)\n",
               name, get_avg(), get_min(), get_max(), count_);
    }

    // The number that tells you whether to keep optimizing. Under ~30% of
    // peak means something structural (coalescing, occupancy); over ~80%
    // means stop. See PERFORMANCE.md.
    void report_bandwidth(const char *name, double bytes) const
    {
        const double achieved = gb_per_s(bytes);
        const double peak = peak_gb_per_s();
        printf("%-24s %8.3f ms  %8.1f GB/s  (%.0f%% of %.0f GB/s peak)\n",
               name, get_avg(), achieved, 100.0 * achieved / peak, peak);
    }

    void report_tflops(const char *name, double flops) const
    {
        const double achieved = tflops(flops);
        const double peak = peak_tflops_fp32();
        printf("%-24s %8.3f ms  %8.2f TFLOP/s  (%.0f%% of %.1f TFLOP/s peak)\n",
               name, get_avg(), achieved, 100.0 * achieved / peak, peak);
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
    double total_ms_   = 0.0;
    double min_ms_     = FLT_MAX;
    double max_ms_     = 0.0;
    int    count_      = 0;
};
