Memory management system
- paged memory (refer to OS mem paging) - malloc, pros, cons
- pinned memory - cudaMallocHost, pros, cons
- page locking (preventing OS to move pages around) - cudaHostRegister, pros, cons
- mapped memory (zero-copy memory) - cudaHostAlloc with cudaHostAllocMapped flag set, pros, cons, 
- unified memory - cudaMallocManaged, pros, cons - __managed__

[https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf](https://developer.codeplay.com/products/computecpp/ce/1.3.0/guides/sycl-for-cuda-developers/memory-model)

Use pinned memory,
Improve prev algorithm using pinned memory and monitor using nsight systems
