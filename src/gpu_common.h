#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <stdbool.h>

// Define GPU runtime API macros
#if defined(__CUDACC__) || defined(__HIPCC__)

// Include appropriate headers based on the compiler
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#define GPU(expr) hip##expr
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuGetLastError() hipGetLastError()
#define gpuGetErrorString(e) hipGetErrorString(e)
#define gpuSetDevice(id) hipSetDevice(id)
#define gpuMalloc(ptr, size) hipMalloc(ptr, size)
#define gpuFree(ptr) hipFree(ptr)
#define gpuMemcpy(dst, src, size, kind) hipMemcpy(dst, src, size, kind)
#define gpuMemcpyAsync(dst, src, size, kind, stream) hipMemcpyAsync(dst, src, size, kind, stream)
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuGetDeviceProperties(prop, id) hipGetDeviceProperties(prop, id)
#define gpuDeviceGetAttribute(pi, attr, device) hipDeviceGetAttribute(pi, (hipDeviceAttribute_t)attr, device)
#define gpuGetDeviceCount(count) hipGetDeviceCount(count)
#define gpuEventCreate(event) hipEventCreate(event)
#define gpuEventDestroy(event) hipEventDestroy(event)
#define gpuEventRecord(event, stream) hipEventRecord(event, stream)
#define gpuEventSynchronize(event) hipEventSynchronize(event)
#define gpuEventElapsedTime(ms, start, end) hipEventElapsedTime(ms, start, end)
#define gpuStreamCreate(stream) hipStreamCreate(stream)
#define gpuStreamDestroy(stream) hipStreamDestroy(stream)
#define gpuStreamSynchronize(stream) hipStreamSynchronize(stream)
#define gpuMemset(dst, val, size) hipMemset(dst, val, size)
#define gpuMemsetAsync(dst, val, size, stream) hipMemsetAsync(dst, val, size, stream)

// RNG
#define curandState hiprandState_t
#define curand_init(seed, seq, offset, state) hiprand_init(seed, seq, 0, state)
#define curand_uniform(state) hiprand_uniform_double(state)

#else // CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define GPU(expr) cuda##expr
typedef cudaError_t gpuError_t;
typedef cudaStream_t gpuStream_t;
typedef cudaEvent_t gpuEvent_t;
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define gpuGetLastError() cudaGetLastError()
#define gpuGetErrorString(e) cudaGetErrorString(e)
#define gpuSetDevice(id) cudaSetDevice(id)
#define gpuMalloc(ptr, size) cudaMalloc(ptr, size)
#define gpuFree(ptr) cudaFree(ptr)
#define gpuMemcpy(dst, src, size, kind) cudaMemcpy(dst, src, size, kind)
#define gpuMemcpyAsync(dst, src, size, kind, stream) cudaMemcpyAsync(dst, src, size, kind, stream)
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuDeviceProp_t cudaDeviceProp
typedef cudaDeviceProp gpuDeviceProp_t;
#define gpuGetDeviceProperties(prop, id) cudaGetDeviceProperties(prop, id)
#define gpuDeviceGetAttribute(pi, attr, device) cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attr, device)
#define gpuGetDeviceCount(count) cudaGetDeviceCount(count)
#define gpuEventCreate(event) cudaEventCreate(event)
#define gpuEventDestroy(event) cudaEventDestroy(event)
#define gpuEventRecord(event, stream) cudaEventRecord(event, stream)
#define gpuEventSynchronize(event) cudaEventSynchronize(event)
#define gpuEventElapsedTime(ms, start, end) cudaEventElapsedTime(ms, start, end)
#define gpuStreamCreate(stream) cudaStreamCreate(stream)
#define gpuStreamDestroy(stream) cudaStreamDestroy(stream)
#define gpuStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define gpuMemset(dst, val, size) cudaMemset(dst, val, size)
#define gpuMemsetAsync(dst, val, size, stream) cudaMemsetAsync(dst, val, size, stream)

// RNG
typedef curandState_t curandState;
#define curand_init(seed, seq, offset, state) curand_init(seed, seq, offset, state)
#define curand_uniform(state) curand_uniform_double(state)

#endif // __HIPCC__

// Common error checking macro
#define GPU_ERROR(ans) \
  do { \
    gpuAssert((ans), __FILE__, __LINE__, true); \
  } while (0)

// Common error checking function
static inline void gpuAssert(gpuError_t code, const char *file, int line, bool abort) {
  if (code != gpuSuccess) {
    fprintf(stderr, "GPUassert: \"%s\" in %s:%d\n", gpuGetErrorString(code), file, line);
    if (abort) {
      exit((int)code);
    }
  }
}

// Kernel launch macros
#ifdef __HIPCC__
#define KERNEL_NAME(name) name##_hip
#else
#define KERNEL_NAME(name) name
#endif

#define KERNEL_LAUNCH(kernel, blocks, threads, shared_mem, stream, ...) \
  kernel<<<blocks, threads, shared_mem, stream>>>(__VA_ARGS__)

#endif // __CUDACC__ || __HIPCC__

#endif // GPU_COMMON_H
