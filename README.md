# CUDA course (Compute Unified Device Architecture)
Title: Fundamentals of Accelerated Computation Using CUDA C/C++

// Oxford course link
https://people.maths.ox.ac.uk/~gilesm/cuda/

// labs
https://iis-people.ee.ethz.ch/~gmichi/asocd_2014/exercises/ex_03.pdf
// lectures from the link

// Nvidia official c programming guide
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf

// Programming model
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model

// Parallel programming with CUDA
http://www.davidmuench.de/studienarbeit.pdf

// 

// Intorduction
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_01.pdf
- tasks, trivial vector addition example

// Different memory and variable types - basic kernel implementation
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_02.pdf

// control flow, atomics
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_03.pdf

// warp based programming model -> too complex after lecture 2, need to do computational tasks
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_04.pdf

// libraries, reomve this topic and provide computational tasks
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_05.pdf

// streams and host related code
- https://people.maths.ox.ac.uk/~gilesm/cuda/2019/lecture_06.pdf

// Usefull links
http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf

// Nvidia lectures
https://developer.nvidia.com/educators/existing-courses#2


// cuda-gdb
set cuda break_on_launch application
cuda device sm warp lane block thread
//step

Exam notes
- kernels and launch
- warp and operations
- shared memory
- paged/pinned memory
- atomic operations and global memory
- mapped memory
- memory transfers, sync/async launch
- streams and events, synchronization
- graph and graph record
- texture memory and binding
- bank conflicts and cache control

Tasks:
- opencv filters and optimizations - remove kernel matrices and provide parameters instead
- Compute Beblid descriptors with cuda - https://github.com/opencv/opencv_contrib/blob/80f1ca2442982ed518076cd88cf08c71155b30f6/modules/xfeatures2d/src/beblid.cpp
- optimze RANSAC taking into account that camera moves toward to the scene (use cuda to solve liear equations)
- given 
1) reference objects T1, T2, .. Tn, 
2) some operator <, that T1 < T2 < ... Tn
3) L1 < L2 < ..... Lm
4) P = {Pij, i = 1, ...m, j = 1, ... m} and Pij is the probability that Li is close to Tj
5) find assignement vector V, such that sum(P(V)) -> min (solve with cuda) 
// comment: this is kuhn muknres problem with restrictions, if L[i] is assigned to T[j], then L[i + 1] can be assigned to T[j+1], ... T[n] only
// sould solved for large Ps, so cuda is needed.
- Transformer TrTr with batches ??
