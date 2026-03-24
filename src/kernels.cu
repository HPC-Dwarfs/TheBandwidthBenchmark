#include "portability.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "cli.h"
#include "constants.h"
#include "timing.h"
#include "util.h"

static int getSharedMemSize(
    int THREAD_BLOCK_SIZE, int thread_blocks_per_sm, const void *func);
static void setBlockSize(void);
}

#define GPU_ERROR(ans)                                                                   \
  do {                                                                                   \
    gpuAssert((ans), __FILE__, __LINE__, true);                                          \
  } while (0)

#define VECTORIZED_VERSION_DISPATCH(KERNEL, THPBLKS, ...)                                \
  switch (VEC_VARIANT) {                                                                 \
  case VEC0: {                                                                           \
    HARNESS((KERNEL<<<N / THPBLKS + 1, THPBLKS, shared_mem_size>>>(__VA_ARGS__)), KERNEL)              \
    break;                                                                               \
  }                                                                                      \
  case VEC2: {                                                                           \
    HARNESS((KERNEL##_vec2<<<(N / 2) / THPBLKS + 1, THPBLKS, shared_mem_size>>>(__VA_ARGS__)),         \
        KERNEL##_vec2)                                                                   \
    break;                                                                               \
  }                                                                                      \
  case VEC4: {                                                                           \
    HARNESS((KERNEL##_vec4<<<(N / 4) / THPBLKS + 1, THPBLKS, shared_mem_size>>>(__VA_ARGS__)),         \
        KERNEL##_vec4)                                                                   \
    break;                                                                               \
  }                                                                                      \
  }                                                                                      \
  printf("something horrible UB is happening and u should never reach this point since " \
         "we pre-check");                                                                \
  abort();

static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: \"%s\" in %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit((int)code);
    }
  }
}

__global__ void init_constants(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  if (a != nullptr) a[tidx] = INIT_A;
  if (b != nullptr) b[tidx] = INIT_B;
  if (c != nullptr) c[tidx] = INIT_C;
  if (d != nullptr) d[tidx] = INIT_D;
}

__global__ void init_randoms(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N,
    unsigned long long seed)
{

  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  // Declare and initialize RNG state
  curandState state;
  curand_init(seed, tidx, 0,
      &state); // seed, sequence number, offset, &state

  if (a != nullptr) a[tidx] = (TBB_FLOAT)curand_uniform(&state);
  if (b != nullptr) b[tidx] = (TBB_FLOAT)curand_uniform(&state);
  if (c != nullptr) c[tidx] = (TBB_FLOAT)curand_uniform(&state);
  if (d != nullptr) d[tidx] = (TBB_FLOAT)curand_uniform(&state);
}

__global__ void initCuda(TBB_FLOAT *__restrict__ b, TBB_FLOAT scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  b[tidx] = scalar;
}

__global__ void initCuda_vec2(TBB_FLOAT *__restrict__ b, TBB_FLOAT scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 2) {
    return;
  }

  TBB_FLOAT2 *b2 = reinterpret_cast<TBB_FLOAT2 *>(b) + tidx;

  b2->x       = scalar;
  b2->y       = scalar;

  if ((N % 2 == 1) && (tidx == N / 2)) {
    b[N - 1] = scalar;
  }
}

__global__ void initCuda_vec4(TBB_FLOAT *__restrict__ b, int scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 4) {
    return;
  }

  TBB_FLOAT4 *b4   = reinterpret_cast<TBB_FLOAT4 *>(b) + tidx;

  b4->x         = scalar;
  b4->y         = scalar;
  b4->z         = scalar;
  b4->w         = scalar;

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      b[idx]  = scalar;
    }
  }
}

__global__ void copyCuda(
    TBB_FLOAT *__restrict__ c, TBB_FLOAT *__restrict__ a, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  c[tidx] = a[tidx];
}

__global__ void copyCuda_vec2(
    TBB_FLOAT *__restrict__ c, TBB_FLOAT *__restrict__ a, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 2) {
    return;
  }

  reinterpret_cast<TBB_FLOAT2 *>(c)[tidx] = reinterpret_cast<TBB_FLOAT2 *>(a)[tidx];

  // in only one thread, process final element (if there is one)
  if (tidx == N / 2 && N % 2 == 1)
    c[N - 1] = a[N - 1];
}

__global__ void copyCuda_vec4(
    TBB_FLOAT *__restrict__ c, TBB_FLOAT *__restrict__ a, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 4) {
    return;
  }

  reinterpret_cast<TBB_FLOAT4 *>(c)[tidx] = reinterpret_cast<TBB_FLOAT4 *>(a)[tidx];

  int remainder                        = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      c[idx]  = a[idx];
    }
  }
}

__global__ void updateCuda(TBB_FLOAT *__restrict__ a, int scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] * scalar;
}

__global__ void updateCuda_vec2(TBB_FLOAT *__restrict__ a, int scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 2) {
    return;
  }

  TBB_FLOAT2 *a2 = reinterpret_cast<TBB_FLOAT2 *>(a) + tidx;

  a2->x *= scalar;
  a2->y *= scalar;

  if ((N % 2 == 1) && (tidx == N / 2)) {
    a[N - 1] *= scalar;
  }
}

__global__ void updateCuda_vec4(TBB_FLOAT *__restrict__ a, int scalar, const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 4) {
    return;
  }

  TBB_FLOAT4 *a2 = reinterpret_cast<TBB_FLOAT4 *>(a) + tidx;

  a2->x *= scalar;
  a2->y *= scalar;
  a2->z *= scalar;
  a2->w *= scalar;

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      a[idx] *= scalar;
    }
  }
}

__global__ void triadCuda(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = b[tidx] + (scalar * c[tidx]);
}

__global__ void triadCuda_vec2(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 2) {
    return;
  }

  TBB_FLOAT2 *a2 = reinterpret_cast<TBB_FLOAT2 *>(a) + tidx;
  TBB_FLOAT2 *b2 = reinterpret_cast<TBB_FLOAT2 *>(b) + tidx;
  TBB_FLOAT2 *c2 = reinterpret_cast<TBB_FLOAT2 *>(c) + tidx;

  a2->x       = b2->x + (scalar * c2->x);
  a2->y       = b2->y + (scalar * c2->y);

  if ((N % 2 == 1) && (tidx == N / 2)) {
    a[N - 1] = b[N - 1] + (scalar * c[N - 1]);
  }
}

__global__ void triadCuda_vec4(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 4) {
    return;
  }

  TBB_FLOAT4 *a4   = reinterpret_cast<TBB_FLOAT4 *>(a) + tidx;
  TBB_FLOAT4 *b4   = reinterpret_cast<TBB_FLOAT4 *>(b) + tidx;
  TBB_FLOAT4 *c4   = reinterpret_cast<TBB_FLOAT4 *>(c) + tidx;

  a4->x         = b4->x + (scalar * c4->x);
  a4->y         = b4->y + (scalar * c4->y);
  a4->z         = b4->z + (scalar * c4->z);
  a4->w         = b4->w + (scalar * c4->w);

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      a[idx]  = b[idx] + (scalar * c[idx]);
    }
  }
}

__global__ void daxpyCuda(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] + (scalar * b[tidx]);
}

__global__ void daxpyCuda_vec2(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 2) {
    return;
  }

  TBB_FLOAT2 *a2 = reinterpret_cast<TBB_FLOAT2 *>(a) + tidx;
  TBB_FLOAT2 *b2 = reinterpret_cast<TBB_FLOAT2 *>(b) + tidx;

  a2->x       = a2->x + (scalar * b2->x);
  a2->y       = a2->y + (scalar * b2->y);

  if ((N % 2 == 1) && (tidx == N / 2)) {
    a[N - 1] = a[N - 1] + (scalar * b[N - 1]);
  }
}

__global__ void daxpyCuda_vec4(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    const int scalar,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N / 4) {
    return;
  }

  TBB_FLOAT4 *a4   = reinterpret_cast<TBB_FLOAT4 *>(a) + tidx;
  TBB_FLOAT4 *b4   = reinterpret_cast<TBB_FLOAT4 *>(b) + tidx;

  a4->x         = a4->x + (scalar * b4->x);
  a4->y         = a4->y + (scalar * b4->y);
  a4->z         = a4->z + (scalar * b4->z);
  a4->w         = a4->w + (scalar * b4->w);

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      a[idx]  = a[idx] + (scalar * b[idx]);
    }
  }
}

__global__ void striadCuda(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = b[tidx] + (d[tidx] * c[tidx]);
}

__global__ void striadCuda_vec2(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  TBB_FLOAT2 *a2 = reinterpret_cast<TBB_FLOAT2 *>(a) + tidx;
  TBB_FLOAT2 *b2 = reinterpret_cast<TBB_FLOAT2 *>(b) + tidx;
  TBB_FLOAT2 *c2 = reinterpret_cast<TBB_FLOAT2 *>(c) + tidx;
  TBB_FLOAT2 *d2 = reinterpret_cast<TBB_FLOAT2 *>(d) + tidx;

  a2->x       = b2->x + (d2->x * c2->x);
  a2->y       = b2->y + (d2->y * c2->y);

  if ((N % 2 == 1) && (tidx == N / 2)) {
    a[N - 1] = b[N - 1] + (d[N - 1] * c[N - 1]);
  }
}

__global__ void striadCuda_vec4(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  TBB_FLOAT4 *a4   = reinterpret_cast<TBB_FLOAT4 *>(a) + tidx;
  TBB_FLOAT4 *b4   = reinterpret_cast<TBB_FLOAT4 *>(b) + tidx;
  TBB_FLOAT4 *c4   = reinterpret_cast<TBB_FLOAT4 *>(c) + tidx;
  TBB_FLOAT4 *d4   = reinterpret_cast<TBB_FLOAT4 *>(d) + tidx;

  a4->x         = b4->x + (d4->x * c4->x);
  a4->y         = b4->y + (d4->y * c4->y);
  a4->z         = b4->z + (d4->z * c4->z);
  a4->w         = b4->w + (d4->w * c4->w);

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      a[idx]  = b[idx] + (d[idx] * c[idx]);
    }
  }
}

__global__ void sdaxpyCuda(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] + (b[tidx] * c[tidx]);
}

__global__ void sdaxpyCuda_vec2(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  TBB_FLOAT2 *a2 = reinterpret_cast<TBB_FLOAT2 *>(a) + tidx;
  TBB_FLOAT2 *b2 = reinterpret_cast<TBB_FLOAT2 *>(b) + tidx;
  TBB_FLOAT2 *c2 = reinterpret_cast<TBB_FLOAT2 *>(c) + tidx;

  a2->x       = a2->x + (b2->x * c2->x);
  a2->y       = a2->y + (b2->y * c2->y);

  if ((N % 2 == 1) && (tidx == N / 2)) {
    a[N - 1] = a[N - 1] + (b[N - 1] * c[N - 1]);
  }
}

__global__ void sdaxpyCuda_vec4(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const size_t N)
{
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  TBB_FLOAT4 *a4   = reinterpret_cast<TBB_FLOAT4 *>(a) + tidx;
  TBB_FLOAT4 *b4   = reinterpret_cast<TBB_FLOAT4 *>(b) + tidx;
  TBB_FLOAT4 *c4   = reinterpret_cast<TBB_FLOAT4 *>(c) + tidx;

  a4->x         = a4->x + (b4->x * c4->x);
  a4->y         = a4->y + (b4->y * c4->y);
  a4->z         = a4->z + (b4->z * c4->z);
  a4->w         = a4->w + (b4->w * c4->w);

  int remainder = N % 4;
  if (tidx == N / 4 && remainder != 0) {
    while (remainder) {
      int idx = N - remainder--;
      a[idx]  = a[idx] + (b[idx] * c[idx]);
    }
  }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile TBB_FLOAT *shared_data, size_t tidx)
{
  if (blockSize >= 64) shared_data[tidx] += shared_data[tidx + 32];
  if (blockSize >= 32) shared_data[tidx] += shared_data[tidx + 16];
  if (blockSize >= 16) shared_data[tidx] += shared_data[tidx + 8];
  if (blockSize >= 8)  shared_data[tidx] += shared_data[tidx + 4];
  if (blockSize >= 4)  shared_data[tidx] += shared_data[tidx + 2];
  if (blockSize >= 2)  shared_data[tidx] += shared_data[tidx + 1];
}

// Inspired by the
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// REDUCTION 6 – Multiple Adds / Threads
template <unsigned int blockSize>
__global__ void sumCuda(
    TBB_FLOAT *__restrict__ a, TBB_FLOAT *__restrict__ a_out, const size_t N)
{
  extern __shared__ TBB_FLOAT shared_data[];

  size_t tidx       = threadIdx.x;
  size_t i          = blockIdx.x * (blockSize * 2) + tidx;
  size_t gridSize   = blockDim.x * 2 * gridDim.x;
  
  shared_data[tidx] = 0;

  while (i < N) {
    shared_data[tidx] += a[i] + a[i + blockSize];
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 1024) {
    if (tidx < 512) { shared_data[tidx] += shared_data[tidx + 512]; }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tidx < 256) { shared_data[tidx] += shared_data[tidx + 256]; }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tidx < 128) { shared_data[tidx] += shared_data[tidx + 128]; }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tidx < 64) { shared_data[tidx] += shared_data[tidx + 64]; }
    __syncthreads();
  }
  if (blockSize >= 64) {
    if (tidx < 32) { shared_data[tidx] += shared_data[tidx + 32]; }
    __syncthreads();
  }

  if (tidx < 32) {
    warpReduce<blockSize>(shared_data, tidx);
  }

  if (tidx == 0) {
    a_out[blockIdx.x] = shared_data[0];
  }
}

__global__ void sumCudaGeneric(
    TBB_FLOAT *__restrict__ a, TBB_FLOAT *__restrict__ a_out, const size_t N)
{
  extern __shared__ TBB_FLOAT shared_data[];

  size_t tidx       = threadIdx.x;
  size_t i          = blockIdx.x * (blockDim.x * 2) + tidx;
  size_t gridSize   = blockDim.x * 2 * gridDim.x;
  
  TBB_FLOAT sum = 0;

  while (i < N) {
    sum += a[i];
    if (i + blockDim.x < N) {
      sum += a[i + blockDim.x];
    }
    i += gridSize;
  }
  shared_data[tidx] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x; s > 1; ) {
    unsigned int half = (s + 1) / 2;
    if (tidx < s / 2) {
      shared_data[tidx] += shared_data[tidx + half];
    }
    s = half;
    __syncthreads();
  }

  if (tidx == 0) {
    a_out[blockIdx.x] = shared_data[0];
  }
}

#define SHARED_MEM(kernel_name)                                                          \
  getSharedMemSize(THREAD_BLOCK_SIZE, THREAD_BLOCK_PER_SM, (const void *)&(kernel_name))

#define HARNESS(kernel, kernel_name)                                                     \
  int shared_mem_size = SHARED_MEM(kernel_name);                                         \
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));                                                 \
  GPU_ERROR(cudaFree(0));                                                                \
  double S = getTimeStamp();                                                             \
  kernel;                                                                                \
  GPU_ERROR(cudaGetLastError());                                                         \
  GPU_ERROR(cudaDeviceSynchronize());                                                    \
  double E = getTimeStamp();                                                             \
  return E - S;

extern "C" {
void allocateArrays(
    TBB_FLOAT **a, TBB_FLOAT **b, TBB_FLOAT **c, TBB_FLOAT **d, const size_t N)
{
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  GPU_ERROR(cudaMalloc((void **)a, N * sizeof(TBB_FLOAT)));
  GPU_ERROR(cudaMalloc((void **)b, N * sizeof(TBB_FLOAT)));
  GPU_ERROR(cudaMalloc((void **)c, N * sizeof(TBB_FLOAT)));
  GPU_ERROR(cudaMalloc((void **)d, N * sizeof(TBB_FLOAT)));
}

void initArrays(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  setBlockSize();

  if (DataInitVariant == CONSTANT) {

    init_constants<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE>>>(a, b, c, d, N);

  } else if (DataInitVariant == RANDOM) {
    printf("Using random initialization.\n");

    unsigned long long seed = time(NULL); // unique seed
    init_randoms<<<(N / THREAD_BLOCK_SIZE) + 1, THREAD_BLOCK_SIZE>>>(a, b, c, d, N, seed);
  }

  GPU_ERROR(cudaDeviceSynchronize());
}

void reinitSweepBuffers(TBB_FLOAT **a, TBB_FLOAT **b, size_t bufferCount)
{
  GPU_ERROR(cudaFree(*a));
  GPU_ERROR(cudaFree(*b));

  GPU_ERROR(cudaMalloc((void **)a, bufferCount * sizeof(TBB_FLOAT)));
  GPU_ERROR(cudaMalloc((void **)b, bufferCount * sizeof(TBB_FLOAT)));

  setBlockSize();

  if (DataInitVariant == CONSTANT) {

    init_constants<<<bufferCount / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE>>>(*a, *b, nullptr, nullptr, bufferCount);

  } else if (DataInitVariant == RANDOM) {

    unsigned long long seed = time(NULL); // unique seed
    init_randoms<<<(bufferCount / THREAD_BLOCK_SIZE) + 1, THREAD_BLOCK_SIZE>>>(*a, *b, nullptr, nullptr, bufferCount, seed);
  }

  GPU_ERROR(cudaDeviceSynchronize());
}

double init(TBB_FLOAT *__restrict__ b, TBB_FLOAT scalar, const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(initCuda, THREAD_BLOCK_SIZE, b, scalar, N);
}

double copy(TBB_FLOAT *__restrict__ c, TBB_FLOAT *__restrict__ a, const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(copyCuda, THREAD_BLOCK_SIZE, c, a, N);
}

double update(TBB_FLOAT *__restrict__ a, TBB_FLOAT scalar, const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(updateCuda, THREAD_BLOCK_SIZE, a, scalar, N);
}

double triad(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const TBB_FLOAT scalar,
    const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(triadCuda, THREAD_BLOCK_SIZE, a, b, c, scalar, N);
}

double daxpy(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    const TBB_FLOAT scalar,
    const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(daxpyCuda, THREAD_BLOCK_SIZE, a, b, scalar, N);
}

double striad(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    TBB_FLOAT *__restrict__ d,
    const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(striadCuda, THREAD_BLOCK_SIZE, a, b, c, d, N);
}

double sdaxpy(TBB_FLOAT *__restrict__ a,
    TBB_FLOAT *__restrict__ b,
    TBB_FLOAT *__restrict__ c,
    const size_t N)
{
  VECTORIZED_VERSION_DISPATCH(sdaxpyCuda, THREAD_BLOCK_SIZE, a, b, c, N);
}

double sum(TBB_FLOAT *__restrict__ a, const size_t N)
{
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  TBB_FLOAT *al;

  GPU_ERROR(cudaMalloc(
      &al, (N + (THREAD_BLOCK_SIZE - 1)) / THREAD_BLOCK_SIZE * sizeof(TBB_FLOAT)));

  double start = getTimeStamp();

  size_t num_blocks = N / (THREAD_BLOCK_SIZE * 2) + 1;

  switch (THREAD_BLOCK_SIZE) {
    case 1024:
      sumCuda<1024><<<num_blocks, 1024, 1024 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    case 512:
      sumCuda<512><<<num_blocks, 512, 512 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    case 256:
      sumCuda<256><<<num_blocks, 256, 256 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    case 128:
      sumCuda<128><<<num_blocks, 128, 128 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    case 64:
      sumCuda<64><<<num_blocks, 64, 64 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    case 32:
      sumCuda<32><<<num_blocks, 32, 32 * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
    default:
      sumCudaGeneric<<<num_blocks, THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE * sizeof(TBB_FLOAT)>>>(a, al, N);
      break;
  }

  GPU_ERROR(cudaDeviceSynchronize());

  double end = getTimeStamp();

  GPU_ERROR(cudaFree(al));

  return end - start;
}

void setBlockSize()
{
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, 0));

  // int max_THREAD_BLOCK_SIZE                    = prop.maxThreadsPerBlock;
  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

  // Not the best case for THREAD_BLOCK_SIZE.
  // Varying THREAD_BLOCK_SIZE can result in
  // better performance and thread occupancy.
  if (THREAD_BLOCK_SIZE_SET == 0) {
    THREAD_BLOCK_SIZE = prop.maxThreadsPerMultiProcessor / 2;
#ifdef THREADBLOCKSIZE
    THREAD_BLOCK_SIZE = THREADBLOCKSIZE;
#endif
  }

  if (THREAD_BLOCK_PER_SM_SET == 0) {
#ifdef THREADBLOCKPERSM
    THREAD_BLOCK_PER_SM = THREADBLOCKPERSM;
#endif
  }

  THREAD_BLOCK_PER_SM =
      MIN(floor(maxThreadsPerSM / THREAD_BLOCK_SIZE), THREAD_BLOCK_PER_SM);

  double occupancy = (((double)THREAD_BLOCK_SIZE * (double)THREAD_BLOCK_PER_SM) /
                         (double)maxThreadsPerSM) *
                     100;
  if(GPUBenchmarkType == GPU_WS) {
    printf(HLINE);
    printf("Thread Block Size: \t %d\n", THREAD_BLOCK_SIZE);
    printf("Thread Block Per SM: \t %d\n", THREAD_BLOCK_PER_SM);
    printf("Occupancy: \t\t %.2f %% \n", occupancy);
  }
  
}

int getSharedMemSize(int THREAD_BLOCK_SIZE, int thread_blocks_per_sm, const void *func)
{
  int max_active_thread_blocks = 0;
  
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_thread_blocks, func, THREAD_BLOCK_SIZE, 1));
      
  if (max_active_thread_blocks <= thread_blocks_per_sm) {
    return 1;
  }

  int shared_mem_size = 1024;
  cudaError_t err = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  if (err != cudaSuccess) {
    cudaGetLastError(); // clear error state
    return 1; // hardware doesn't support this, can't limit occupancy
  }

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_thread_blocks, func, THREAD_BLOCK_SIZE, shared_mem_size));

  while (max_active_thread_blocks > thread_blocks_per_sm) {
    shared_mem_size += 256;
    err = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    if (err != cudaSuccess) {
      cudaGetLastError(); // clear error state
      shared_mem_size -= 256; // revert to the highest successful size
      break;
    }
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_thread_blocks, func, THREAD_BLOCK_SIZE, shared_mem_size));
  }
  
  // Re-assert final successful size limits before return
  cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  return shared_mem_size;
}
}