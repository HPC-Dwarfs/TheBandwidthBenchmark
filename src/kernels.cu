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

static int getSharedMemSize(int THREAD_BLOCK_SIZE, int thread_blocks_per_sm,
                            const void *func);
static void setBlockSize(void);
}

#define GPU_ERROR(ans)                                                         \
  do {                                                                         \
    gpuAssert((ans), __FILE__, __LINE__, true);                                \
  } while (0)

static inline void gpuAssert(cudaError_t code, const char *file, int line,
                             bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: \"%s\" in %s:%d\n", cudaGetErrorString(code),
            file, line);
    if (abort) {
      exit((int)code);
    }
  }
}

__global__ void init_constants(double *__restrict__ a, double *__restrict__ b,
                               double *__restrict__ c, double *__restrict__ d,
                               const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = INIT_A;
  b[tidx] = INIT_B;
  c[tidx] = INIT_C;
  d[tidx] = INIT_D;
}

__global__ void init_randoms(double *__restrict__ a, double *__restrict__ b,
                             double *__restrict__ c, double *__restrict__ d,
                             const size_t N, unsigned long long seed) {

  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  // Declare and initialize RNG state
  curandState state;
  curand_init(seed, tidx, 0,
              &state); // seed, sequence number, offset, &state

  a[tidx] = (double)curand_uniform(&state);
  b[tidx] = (double)curand_uniform(&state);
  c[tidx] = (double)curand_uniform(&state);
  d[tidx] = (double)curand_uniform(&state);
}

__global__ void initCuda(double *__restrict__ b, int scalar, const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  b[tidx] = scalar;
}

__global__ void copyCuda(double *__restrict__ c, double *__restrict__ a,
                         const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  c[tidx] = a[tidx];
}

__global__ void updateCuda(double *__restrict__ a, int scalar, const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] * scalar;
}

__global__ void triadCuda(double *__restrict__ a, double *__restrict__ b,
                          double *__restrict__ c, const int scalar,
                          const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = b[tidx] + (scalar * c[tidx]);
}

__global__ void daxpyCuda(double *__restrict__ a, double *__restrict__ b,
                          const int scalar, const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] + (scalar * b[tidx]);
}

__global__ void striadCuda(double *__restrict__ a, double *__restrict__ b,
                           double *__restrict__ c, double *__restrict__ d,
                           const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = b[tidx] + (d[tidx] * c[tidx]);
}

__global__ void sdaxpyCuda(double *__restrict__ a, double *__restrict__ b,
                           double *__restrict__ c, const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= N) {
    return;
  }

  a[tidx] = a[tidx] + (b[tidx] * c[tidx]);
}

__device__ void warpReduce(volatile int *shared_data, size_t tidx) {
  shared_data[tidx] += shared_data[tidx + 32];
  shared_data[tidx] += shared_data[tidx + 16];
  shared_data[tidx] += shared_data[tidx + 8];
  shared_data[tidx] += shared_data[tidx + 4];
  shared_data[tidx] += shared_data[tidx + 2];
  shared_data[tidx] += shared_data[tidx + 1];
}

// Inspired by the
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__global__ void sumCuda(double *__restrict__ a, double *__restrict__ a_out,
                        const size_t N) {
  extern __shared__ int shared_data[];

  size_t tidx = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  shared_data[tidx] = a[i] + a[i + blockDim.x];
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {

    if (tidx < s) {
      shared_data[tidx] += shared_data[tidx + s];
    }
    __syncthreads();
  }

  if (tidx < 32) {
    warpReduce(shared_data, tidx);
  }

  if (tidx == 0) {
    a[blockIdx.x] = shared_data[0];
  }
}

#define SHARED_MEM(kernel_name)                                                \
  getSharedMemSize(THREAD_BLOCK_SIZE, THREAD_BLOCK_PER_SM,                     \
                   (const void *)&(kernel_name))

#define HARNESS(kernel, kernel_name)                                           \
  int shared_mem_size = SHARED_MEM(kernel_name);                               \
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));                                       \
  GPU_ERROR(cudaFree(0));                                                      \
  double S = getTimeStamp();                                                   \
  kernel;                                                                      \
  GPU_ERROR(cudaDeviceSynchronize());                                          \
  double E = getTimeStamp();                                                   \
  return E - S;

extern "C" {
void allocateArrays(double **a, double **b, double **c, double **d,
                    const size_t N) {
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  GPU_ERROR(cudaMalloc((void **)a, N * sizeof(double)));
  GPU_ERROR(cudaMalloc((void **)b, N * sizeof(double)));
  GPU_ERROR(cudaMalloc((void **)c, N * sizeof(double)));
  GPU_ERROR(cudaMalloc((void **)d, N * sizeof(double)));
}

void initArrays(double *__restrict__ a, double *__restrict__ b,
                double *__restrict__ c, double *__restrict__ d,
                const size_t N) {
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  setBlockSize();

  if (DataInitVariant == CONSTANT) {

    init_constants<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE>>>(a, b, c, d,
                                                                     N);

  } else if (DataInitVariant == RANDOM) {
    printf("Using random initialization.\n");

    unsigned long long seed = time(NULL); // unique seed
    init_randoms<<<(N / THREAD_BLOCK_SIZE) + 1, THREAD_BLOCK_SIZE>>>(a, b, c, d,
                                                                     N, seed);
  }

  GPU_ERROR(cudaDeviceSynchronize());
}

double init(double *__restrict__ b, double scalar, const size_t N) {
  HARNESS((initCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE, 0>>>(
              b, scalar, N)),
          initCuda)
}

double copy(double *__restrict__ c, double *__restrict__ a, const size_t N) {
  HARNESS((copyCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                      shared_mem_size>>>(c, a, N)),
          copyCuda)
}

double update(double *__restrict__ a, double scalar, const size_t N) {
  HARNESS((updateCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                        shared_mem_size>>>(a, scalar, N)),
          updateCuda)
}

double triad(double *__restrict__ a, double *__restrict__ b,
             double *__restrict__ c, const double scalar, const size_t N) {
  HARNESS((triadCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                       shared_mem_size>>>(a, b, c, scalar, N)),
          triadCuda)
}

double daxpy(double *__restrict__ a, double *__restrict__ b,
             const double scalar, const size_t N) {
  HARNESS((daxpyCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                       shared_mem_size>>>(a, b, scalar, N)),
          daxpyCuda)
}

double striad(double *__restrict__ a, double *__restrict__ b,
              double *__restrict__ c, double *__restrict__ d, const size_t N) {
  HARNESS((striadCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                        shared_mem_size>>>(a, b, c, d, N)),
          striadCuda)
}

double sdaxpy(double *__restrict__ a, double *__restrict__ b,
              double *__restrict__ c, const size_t N) {
  HARNESS((sdaxpyCuda<<<N / THREAD_BLOCK_SIZE + 1, THREAD_BLOCK_SIZE,
                        shared_mem_size>>>(a, b, c, N)),
          sdaxpyCuda)
}

double sum(double *__restrict__ a, const size_t N) {
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaFree(0));

  double *al;

  GPU_ERROR(cudaMalloc(&al, (N + (THREAD_BLOCK_SIZE - 1)) / THREAD_BLOCK_SIZE *
                                sizeof(double)));

  double start = getTimeStamp();

  sumCuda<<<N / (THREAD_BLOCK_SIZE * 2) + 1, THREAD_BLOCK_SIZE,
            THREAD_BLOCK_SIZE * sizeof(double)>>>(a, al, N);

  GPU_ERROR(cudaDeviceSynchronize());

  double end = getTimeStamp();

  GPU_ERROR(cudaFree(al));

  return end - start;
}

void setBlockSize() {
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, 0));

  // int max_THREAD_BLOCK_SIZE                    = prop.maxThreadsPerBlock;
  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

  // Not the best case for THREAD_BLOCK_SIZE.
  // Varying THREAD_BLOCK_SIZE can result in
  // better performance and thread occupancy.
  if (THREAD_BLOCK_SIZE_SET == 0) {
    THREAD_BLOCK_SIZE = prop.maxThreadsPerMultiProcessor / 2;
  }

#ifdef THREADBLOCKSIZE
  THREAD_BLOCK_SIZE = THREADBLOCKSIZE;
#endif

  THREAD_BLOCK_PER_SM =
      MIN(floor(maxThreadsPerSM / THREAD_BLOCK_SIZE), THREAD_BLOCK_PER_SM);

#ifdef THREADBLOCKPERSM
  THREAD_BLOCK_PER_SM = MIN(THREAD_BLOCK_PER_SM, THREADBLOCKPERSM);
#endif

  double occupancy =
      (((double)THREAD_BLOCK_SIZE * (double)THREAD_BLOCK_PER_SM) /
       (double)maxThreadsPerSM) *
      100;

  printf(HLINE);
  printf("Thread Block Size: \t %d\n", THREAD_BLOCK_SIZE);
  printf("Thread Block Per SM: \t %d\n", THREAD_BLOCK_PER_SM);
  printf("Occupancy: \t\t %.2f %% \n", occupancy);
}

int getSharedMemSize(int THREAD_BLOCK_SIZE, int thread_blocks_per_sm,
                     const void *func) {

#ifdef THREADBLOCKPERSM
  int max_active_thread_blocks = 0;
  int shared_mem_size = 1024;

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_thread_blocks, func, THREAD_BLOCK_SIZE, shared_mem_size));

  while (max_active_thread_blocks > thread_blocks_per_sm) {
    shared_mem_size += 256;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_thread_blocks, func, THREAD_BLOCK_SIZE, shared_mem_size));
  }
  return shared_mem_size;
#else
  return 1;
#endif
}
}