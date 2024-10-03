Memory management system
- paged memory (refer to OS mem paging) - malloc, pros, cons
- pinned memory - cudaMallocHost, pros, cons
- page locking (preventing OS to move pages around) - cudaHostRegister, pros, cons
- mapped memory (zero-copy memory) - cudaHostAlloc with cudaHostAllocMapped flag set, pros, cons, 
- unified memory - cudaMallocManaged, pros, cons - __managed__

https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf
