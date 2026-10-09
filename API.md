# API Index · Ֆունկցիաների ցանկ

Every function interface this course uses, grouped by the day that introduces it. Each day's `README.md` carries the signatures themselves in its **Functions · Ֆունկցիաներ** section, with a one-line comment per function — this page is the map, so nothing has to be kept in sync in two places.

Signatures verified against the **CUDA Toolkit 13.3** [Runtime API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/). For concepts see [GLOSSARY.md](GLOSSARY.md); for device-side intrinsics in depth, [INTRINSICS.md](INTRINSICS.md); for which optimization to reach for, [PERFORMANCE.md](PERFORMANCE.md).

---

## By day

| Day | Functions introduced |
|---|---|
| [1](day01/README.md#functions--ֆունկցիաներ) | `cudaMalloc` · `cudaFree` · `cudaMemcpy` · `cudaDeviceSynchronize` · `cudaGetLastError` · `cudaGetDeviceProperties` · device `printf` · `report_device_capabilities` |
| [2](day02/README.md#functions--ֆունկցիաներ) | `__syncthreads` · `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (+`WithFlags`) · `cudaOccupancyMaxPotentialBlockSize` |
| [3](day03/README.md#functions--ֆունկցիաներ) | `__syncwarp` |
| [4](day04/README.md#functions--ֆունկցիաներ) | `cudaHostAlloc` · `cudaFreeHost` · `cudaHostRegister` · `cudaHostUnregister` · `cudaHostGetDevicePointer` · `cudaMallocManaged` · `cudaMemPrefetchAsync` · `cudaMemAdvise` · `cudaMemcpyAsync` |
| [5](day05/README.md#functions--ֆունկցիաներ) | `cudaMallocPitch` · `cudaMemcpy2D` · `__syncthreads` · `__syncwarp` |
| [6](day06/README.md#functions--ֆունկցիաներ) | `cudaStreamCreate` · `cudaEventCreate` / `Record` / `Synchronize` / `ElapsedTime` · `cudaStreamWaitEvent` · `cudaLaunchHostFunc` · `cudaSetDeviceFlags` / `cudaGetDeviceFlags` |
| [7](day07/README.md#functions--ֆունկցիաներ) | `cudaHostAlloc` · `cudaMemcpy` · `cudaMemcpyAsync` |
| [8](day08/README.md#functions--ֆունկցիաներ) | `__shfl_sync` · `__shfl_up_sync` · `__shfl_down_sync` · `__shfl_xor_sync` · `__ballot_sync` · `__popc` |
| [9](day09/README.md#functions--ֆունկցիաներ) | `__ballot_sync` · `__activemask` · `atomicAdd` (+`_block`) · `atomicCAS` · `atomicMax` · `cudaMemset` |
| [10](day10/README.md#functions) | `__popc` / `__popcll` · `cv::cuda::ORB::create` · `detectAndCompute` |
| [11](day11/README.md#functions) | `cudaCreateTextureObject` · `cudaDestroyTextureObject` · `cudaCreateChannelDesc` · `tex1D` / `tex2D` / `tex3D` · `surf2Dread` / `surf2Dwrite` |
| [12](day12/README.md#functions--ֆունկցիաներ) | `cudaStreamBeginCapture` · `cudaStreamEndCapture` · `cudaGraphInstantiate` · `cudaGraphLaunch` |
| [13](day13/README.md#functions--ֆունկցիաներ) | `__ldg` · `__ldcs` · `__ldlu` · `__stcs` |
| [14](day14/README.md#functions--ֆունկցիաներ) | `cublasCreate` / `Destroy` / `SetMathMode` · `cublasSgemm` / `Dgemm` · `nppiFilterBoxBorder_8u_C1R` · `cudaLaunchCooperativeKernel` · `cg::tiled_partition` |
| [15](day15/README.md#functions) | `cudaMallocAsync` · `cudaFreeAsync` · `cudaDeviceGetDefaultMemPool` · `cudaMallocFromPoolAsync` · `cudaMemPoolCreate` / `Destroy` / `SetAttribute` / `GetAttribute` / `TrimTo` / `SetAccess` |

## By topic

| Topic | Day |
|---|---|
| Device query and properties | [1](day01/README.md#functions--ֆունկցիաներ) |
| Error handling | [1](day01/README.md#functions--ֆունկցիաներ) |
| Device memory, linear | [1](day01/README.md#functions--ֆունկցիաներ) |
| Device memory, pitched 2D | [5](day05/README.md#functions--ֆունկցիաներ) |
| Host memory: pinned, mapped, registered | [4](day04/README.md#functions--ֆունկցիաներ) |
| Unified / managed memory, prefetch, advise | [4](day04/README.md#functions--ֆունկցիաներ) |
| Stream-ordered allocation and memory pools | [15](day15/README.md#functions) |
| Copies and fills | [1](day01/README.md#functions--ֆունկցիաներ), [4](day04/README.md#functions--ֆունկցիաներ), [5](day05/README.md#functions--ֆունկցիաներ), [7](day07/README.md#functions--ֆունկցիաներ) |
| Streams | [6](day06/README.md#functions--ֆունկցիաներ) |
| Events and device-side timing | [6](day06/README.md#functions--ֆունկցիաներ) |
| Host functions in a stream | [6](day06/README.md#functions--ֆունկցիաներ) |
| Host-side wait policy (spin / yield / blocking) | [6](day06/README.md#functions--ֆունկցիաներ) |
| Occupancy | [2](day02/README.md#functions--ֆունկցիաներ) |
| Barriers | [2](day02/README.md#functions--ֆունկցիաներ), [3](day03/README.md#functions--ֆունկցիաներ), [5](day05/README.md#functions--ֆունկցիաներ) |
| Warp shuffle | [8](day08/README.md#functions--ֆունկցիաներ) |
| Warp vote and bit counting | [8](day08/README.md#functions--ֆունկցիաներ), [9](day09/README.md#functions--ֆունկցիաներ) |
| Atomics | [9](day09/README.md#functions--ֆունկցիաներ) |
| Cache operators | [13](day13/README.md#functions--ֆունկցիաներ) |
| Textures and surfaces | [11](day11/README.md#functions) |
| CUDA graphs | [12](day12/README.md#functions--ֆունկցիաներ) |
| Libraries: cuBLAS, NPP | [14](day14/README.md#functions--ֆունկցիաներ) |
| Cooperative groups | [14](day14/README.md#functions--ֆունկցիաներ) |

---

## Things not tied to one day

### Launch syntax and device-side built-ins

```c++
my_kernel<<<grid, block>>>(args...);                       // 1. grid, 2. block
my_kernel<<<grid, block, sharedBytes>>>(args...);          // 3. dynamic __shared__ bytes
my_kernel<<<grid, block, sharedBytes, stream>>>(args...);  // 4. stream

__global__  __device__  __host__  __shared__  __constant__  __managed__  __restrict__

threadIdx.{x,y,z}   blockIdx.{x,y,z}   blockDim.{x,y,z}   gridDim.{x,y,z}   warpSize

int i      = blockIdx.x * blockDim.x + threadIdx.x;   // global index
int stride = gridDim.x * blockDim.x;                  // grid-stride loop step
```

### Enum values worth remembering

| Enum | Values |
|---|---|
| `cudaMemcpyKind` | `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, `cudaMemcpyHostToHost`, `cudaMemcpyDefault` |
| `cudaHostAlloc` flags | `cudaHostAllocDefault`, `cudaHostAllocMapped`, `cudaHostAllocPortable`, `cudaHostAllocWriteCombined` |
| `cudaHostRegister` flags | `cudaHostRegisterDefault`, `cudaHostRegisterMapped`, `cudaHostRegisterPortable`, `cudaHostRegisterReadOnly` |
| `cudaMemLocationType` | `cudaMemLocationTypeDevice`, `cudaMemLocationTypeHost`, `cudaMemLocationTypeHostNuma` |
| `cudaMemoryAdvise` | `cudaMemAdviseSetReadMostly`, `…SetPreferredLocation`, `…SetAccessedBy` (+ `Unset…` for each) |
| `cudaStreamCaptureMode` | `cudaStreamCaptureModeGlobal`, `…ThreadLocal`, `…Relaxed` |
| `cudaEvent` flags | `cudaEventDefault`, `cudaEventBlockingSync`, `cudaEventDisableTiming`, `cudaEventInterprocess` |
| `cudaSetDeviceFlags` | `cudaDeviceScheduleAuto`, `cudaDeviceScheduleSpin`, `cudaDeviceScheduleYield`, `cudaDeviceScheduleBlockingSync`, `cudaDeviceMapHost`, `cudaDeviceLmemResizeToMax` |
| `cudaMemPoolAttr` | `cudaMemPoolAttrReleaseThreshold`, `…ReservedMemCurrent`/`High`, `…UsedMemCurrent`/`High` — and the reuse policies **without** the `Attr` infix: `cudaMemPoolReuseFollowEventDependencies`, `cudaMemPoolReuseAllowOpportunistic`, `cudaMemPoolReuseAllowInternalDependencies` |
| `cudaFilterMode` | `cudaFilterModePoint`, `cudaFilterModeLinear` |
| `cudaTextureAddressMode` | `cudaAddressModeWrap`, `…Clamp`, `…Mirror`, `…Border` |
| `cublasOperation_t` | `CUBLAS_OP_N`, `CUBLAS_OP_T`, `CUBLAS_OP_C` |

Useful `cudaDeviceAttr` for `cudaDeviceGetAttribute` — prefer these over the matching `cudaDeviceProp` fields for clocks, which are deprecated as of CUDA 12:

`cudaDevAttrClockRate` (kHz) · `cudaDevAttrMemoryClockRate` (kHz) · `cudaDevAttrGlobalMemoryBusWidth` (bits) · `cudaDevAttrMultiProcessorCount` · `cudaDevAttrComputeCapabilityMajor` / `…Minor` · `cudaDevAttrMaxSharedMemoryPerMultiprocessor` · `cudaDevAttrMemoryPoolsSupported` · `cudaDevAttrPageableMemoryAccess`

### OpenCV CUDA essentials

Used from Day 5 onward. Compile with `` `pkg-config --cflags --libs opencv4` ``.

```c++
cv::Mat  cv::imread  ( const String &filename, int flags = cv::IMREAD_COLOR );
void     cv::imshow  ( const String &winname, cv::InputArray mat );
int      cv::waitKey ( int delay = 0 );          // 0 waits for a key; 1 inside a video loop

// cv::cuda::GpuMat
void   upload   ( cv::InputArray arr );
void   download ( cv::OutputArray dst );
void   create   ( int rows, int cols, int type );   // ROWS first — opposite order from dim3
template <class T> T *ptr ( int y = 0 );
size_t step;                                        // row stride IN BYTES — the pitch
int    rows, cols;   cv::Size size();   int type();   bool empty();

int cv::cudev::divUp ( int total, int grain );      // ceil(total / grain)
```

### This course's own helpers

```c++
#include "../common/cuda_check.h"
CUDA_CHECK( call );               // wraps any cudaError_t; prints file/line and exits
CUDA_CHECK_LAST_ERROR();          // call right after every kernel launch

#include "../common/device_info.h"
std::string report_device_capabilities();     // prints the full report, returns the device name

#include "../common/timer.h"
kernel_timer_t t;
void   t.start(cudaStream_t = 0);   void t.stop(cudaStream_t = 0);
double t.get_avg();  t.get_min();  t.get_max();            // milliseconds
double t.gb_per_s(double bytes);
double t.tflops(double flops);      double t.tflops_matmul(int M);
static double kernel_timer_t::peak_gb_per_s(int device = 0);
static double kernel_timer_t::peak_tflops_fp32(int device = 0);
void   t.report(const char *name);
void   t.report_bandwidth(const char *name, double bytes);
void   t.report_tflops(const char *name, double flops);
```

### nvcc and tools

```bash
nvcc -arch=sm_75 template.cu -o day01        # this course's floor; -arch=native for local builds
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_86,code=compute_86 ...      # fat binary + PTX fallback

-Xptxas -v        # registers, shared memory and spills per kernel
-lineinfo         # source lines for the profiler, without disabling optimization
-G                # full device debug (disables optimization)
-maxrregcount=N   # cap registers per thread
--use_fast_math   # route transcendentals through the SFU
-fmad=false       # stop fusing multiply+add into FFMA
--keep            # keep .ptx / .cubin intermediates
-lcurand -lcublas -lcufft -lnppif

cuobjdump --dump-ptx  day01          cuobjdump --dump-sass day01
compute-sanitizer ./day01                       # memcheck: out-of-bounds, misaligned
compute-sanitizer --tool racecheck ./day05      # shared-memory races
compute-sanitizer --tool synccheck ./day08      # illegal barrier / mask use
nsys profile -o report ./day07                  # timeline, whole application
ncu --set full -o report ./day13                # one kernel, in depth
```
