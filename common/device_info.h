// common/device_info.h
//
// A GPU capability report. Run this once at the very start of Day 1 to
// replace the abstract "GPU architecture fundamentals" objective with real
// numbers for the GPU you're actually running on. Nearly every constraint
// you'll bump into in later days -- max threads per block (Day 2), warp
// size (Day 3), shared memory per block/SM (Day 5, Day 13), tensor cores
// (Day 14), memory bandwidth (throughout) -- is printed right here.
#pragma once

#include <cstdio>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_check.h"

// Maps a compute capability (major.minor) to its architecture codename.
// Not exhaustive -- extend as new architectures ship.
inline std::string arch(int major, int minor)
{
    if (major == 3) return "Kepler";
    if (major == 5) return "Maxwell";
    if (major == 6) return "Pascal";
    if (major == 7 && minor == 0) return "Volta";
    if (major == 7 && minor == 5) return "Turing";
    if (major == 8 && minor == 9) return "Ada Lovelace";
    if (major == 8) return "Ampere";
    if (major == 9) return "Hopper";
    if (major == 10 || major == 12) return "Blackwell";
    return "Unknown (compute capability " + std::to_string(major) + "." + std::to_string(minor) + ")";
}

// Human-readable byte count, e.g. UBytes(4194304) -> "4.00 MB".
inline std::string UBytes(size_t bytes)
{
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", value, units[unit]);
    return std::string(buf);
}

// Prints a detailed report of device 0's capabilities to stderr, and
// returns its name.
inline std::string report_device_capabilities()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cerr << "Total CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "Using device: " << 0 << " of " << deviceCount << std::endl;
    std::cerr << "Device name: " << prop.name << std::endl;

    const auto name = arch(prop.major, prop.minor);
    std::cerr << "Architecture: " << name << std::endl;
    std::cerr << "L2 Cache Size: " << UBytes(prop.l2CacheSize) << std::endl;

    int sharedMemoryPerSM = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    std::cerr << "L1 Cache Size (Shared Memory per SM): " << UBytes(sharedMemoryPerSM) << std::endl;
    std::cerr << "CUDA shared mem per block: " << UBytes(prop.sharedMemPerBlock) << std::endl;
    std::cerr << "CUDA registers per block: " << prop.regsPerBlock << std::endl;
    std::cerr << "CUDA registers per SM: " << prop.regsPerMultiprocessor << std::endl;

    // Compute capabilities
    std::cerr << "Warp size: " << prop.warpSize << std::endl;
    std::cerr << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cerr << "Max block dimensions: [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cerr << "Max grid dimensions: [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cerr << "CUDA total const memory: " << UBytes(prop.totalConstMem) << std::endl;
    std::cerr << "CUDA compute mode: " << prop.computeMode << std::endl;
    std::cerr << "CUDA async engines: " << prop.asyncEngineCount << std::endl;
    std::cerr << "CUDA persist cache size: " << UBytes(prop.persistingL2CacheMaxSize) << std::endl;
    std::cerr << "CUDA max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cerr << "CUDA max warps per SM: " << prop.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cerr << "CUDA global L1: " << (prop.globalL1CacheSupported ? "Yes" : "No") << std::endl;
    std::cerr << "CUDA local L1: " << (prop.localL1CacheSupported ? "Yes" : "No") << std::endl;
    std::cerr << "CUDA float/double perf ratio: " << prop.singleToDoublePrecisionPerfRatio << std::endl;
    std::cerr << "CUDA clock rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    std::cerr << "CUDA memory clock rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cerr << "CUDA bus width: " << prop.memoryBusWidth / 8 << " bytes" << std::endl;
    std::cerr << "CUDA reserved shared mem per block: " << UBytes(prop.reservedSharedMemPerBlock) << std::endl;
    std::cerr << "CUDA concurrent kernels: " << prop.concurrentKernels << std::endl;

    // Advanced features
    std::cerr << "ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cerr << "Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;

    int streamPriorities = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&streamPriorities, cudaDevAttrStreamPrioritiesSupported, 0));
    std::cerr << "Stream priorities supported: " << (streamPriorities ? "Yes" : "No") << std::endl;

    std::cerr << "Using " << prop.multiProcessorCount << " SMs, compute capability " << prop.major << "."
              << prop.minor << std::endl;

    // Tensor Core count per SM (approximate -- NVIDIA doesn't expose this
    // via cudaDeviceProp directly, so these are known-generation values).
    int tensorCoresPerSM = 0;
    if (prop.major >= 7) { // Volta and newer
        if (prop.major == 7) tensorCoresPerSM = 8;       // Volta / Turing
        else if (prop.major == 8) tensorCoresPerSM = 4;  // Ampere / Ada
        else if (prop.major == 9) tensorCoresPerSM = 4;  // Hopper
        else if (prop.major == 10 || prop.major == 12) tensorCoresPerSM = 4; // Blackwell

        if (tensorCoresPerSM > 0) {
            std::cerr << "Tensor cores per SM: " << tensorCoresPerSM << std::endl;
            std::cerr << "Total tensor cores: " << prop.multiProcessorCount * tensorCoresPerSM << std::endl;
        }
    } else {
        std::cerr << "Tensor cores: none (pre-Volta architecture)" << std::endl;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cerr << "Total GPU memory: " << UBytes(total_bytes) << std::endl;
    std::cerr << "Free GPU memory: " << UBytes(free_bytes) << std::endl;
    std::cerr << "Memory usage: " << UBytes(total_bytes - free_bytes) << " / " << UBytes(total_bytes) << std::endl;

    const auto memory_clock_rate_mhz = prop.memoryClockRate / 1000.0;
    const auto bus_width_bytes = prop.memoryBusWidth / 8;
    const auto memory_bandwidth_gbps = 2.0 * memory_clock_rate_mhz * bus_width_bytes / 1000.0;
    std::cerr << "Theoretical memory bandwidth: " << memory_bandwidth_gbps << " GB/s" << std::endl;

    return prop.name;
}
