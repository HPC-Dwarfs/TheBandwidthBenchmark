#ifndef PORTABILITY_H
#define PORTABILITY_H

#if defined(_HIP)

#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice

// On AMD/HIP, device enumeration order (HIP ID) can differ from the
// canonical GPU ID (sorted ascending by PCI BDF: domain/bus/device).
// This function accepts a logical gpu_id in BDF-sorted order and maps
// it to the corresponding HIP device index before calling hipSetDevice.
static inline hipError_t hipSetDeviceByGpuId(int gpu_id) {
  int num_devices = 0;
  hipError_t err = hipGetDeviceCount(&num_devices);
  if (err != hipSuccess) return err;

  int bdf_keys[64], hip_indices[64];
  int count = (num_devices < 64) ? num_devices : 64;
  for (int i = 0; i < count; i++) {
    hipDeviceProp_t p;
    err = hipGetDeviceProperties(&p, i);
    if (err != hipSuccess) return err;
    bdf_keys[i]   = (p.pciDomainID << 16) | (p.pciBusID << 8) | p.pciDeviceID;
    hip_indices[i] = i;
  }
  // Insertion sort by BDF key to obtain stable GPU-ID ordering.
  for (int i = 1; i < count; i++) {
    int kb = bdf_keys[i], ki = hip_indices[i];
    int j = i - 1;
    while (j >= 0 && bdf_keys[j] > kb) {
      bdf_keys[j + 1]    = bdf_keys[j];
      hip_indices[j + 1] = hip_indices[j];
      j--;
    }
    bdf_keys[j + 1]    = kb;
    hip_indices[j + 1] = ki;
  }

  if (gpu_id < 0 || gpu_id >= count) return hipErrorInvalidDevice;
  return hipSetDevice(hip_indices[gpu_id]);
}

#define cudaSetDevice hipSetDeviceByGpuId

#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor                                    \
  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaGetLastError hipGetLastError
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute hipFuncSetAttribute
#define curandState hiprandState
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform

#else

#include <cuda_runtime.h>
#include <curand_kernel.h>

#endif

#endif // PORTABILITY_H