/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#ifndef PORTABLE_GPU_H_
#define PORTABLE_GPU_H_

/*
 * GPU Portability Layer
 *
 * This header provides a unified interface for GPU programming that works
 * with both NVIDIA CUDA and AMD HIP toolchains. It uses compile-time
 * detection to map CUDA APIs to their HIP equivalents when building with
 * the HIP compiler.
 */

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) ||          \
    defined(__HIPCC__)
/* HIP toolchain detected - use HIP headers and APIs */
#define GPU_HIP 1
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

/* Runtime API type mappings */
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString

/* Device management */
#define gpuSetDevice hipSetDevice
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp hipDeviceProp_t
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
  hipOccupancyMaxActiveBlocksPerMultiprocessor

/* Memory management */
#define gpuMalloc hipMalloc
#define gpuFree hipFree

/* Random number generation */
#define gpurandState hiprandState
#define gpurand_init hiprand_init
#define gpurand_uniform hiprand_uniform

#else
/* CUDA toolchain - use CUDA headers and APIs */
#define GPU_CUDA 1
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* Runtime API type mappings */
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString

/* Device management */
#define gpuSetDevice cudaSetDevice
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp cudaDeviceProp
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor

/* Memory management */
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree

/* Random number generation */
#define gpurandState curandState
#define gpurand_init curand_init
#define gpurand_uniform curand_uniform

#endif

#endif /* PORTABLE_GPU_H_ */
