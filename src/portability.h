#ifndef PORTABILITY_H
#define PORTABILITY_H

#if defined(_HIP)

#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaGetErrorString hipGetErrorString
#define cudaSetDevice hipSetDevice
#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor                          \
  hipOccupancyMaxActiveBlocksPerMultiprocessor

#define curandState hiprandState
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform

#else

#include <cuda_runtime.h>
#include <curand_kernel.h>

#endif

#endif // PORTABILITY_H
