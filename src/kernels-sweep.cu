#include "portability.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "cli.h"
#include "constants.h"
#include "kernels.h"
#include "profiler.h"
#include "timing.h"
#include "util.h"
}

#ifdef THREADBLOCKSIZE
#define GPU_SWEEP_BLOCKSIZE THREADBLOCKSIZE
#else
#define GPU_SWEEP_BLOCKSIZE 512
#endif

#define GPU_ERROR(ans)                                                                   \
  do {                                                                                   \
    gpuAssert((ans), __FILE__, __LINE__, true);                                          \
  } while (0)

static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: \"%s\" in %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit((int)code);
    }
  }
}

/**
 * @brief Throughput-Optimized Micro-benchmark Kernel for GPU cache
 *
 * @tparam N         The number of elements to process in the inner loop.
 * @tparam iters     Number of outer loop repetitions.
 * @tparam BLOCKSIZE The stride/offset used for memory access, usually blockDim.x.
 */
template <size_t N, int iters, int BLOCKSIZE>
__global__ void l1Kernel(TBB_FLOAT *__restrict__ a, const TBB_FLOAT *__restrict__ b,
                          int zero) {
    TBB_FLOAT localSum = (TBB_FLOAT)0;

    b+= threadIdx.x;

#pragma unroll ((N / BLOCKSIZE) > 32) ? 1 : (32 / ((N / BLOCKSIZE) > 0 ? (N / BLOCKSIZE) : 1))
    for (int iter = 0; iter < iters; iter++) {
        b+= zero;
        auto b2 = b + N;

#pragma unroll ((N / BLOCKSIZE) >= 64) ? 32 : ((N / BLOCKSIZE) > 0 ? (N / BLOCKSIZE) : 1)
        for (size_t i = 0; i < N; i += BLOCKSIZE) {
            localSum += b[i] * b2[i];
        }

        localSum *= (TBB_FLOAT)1.3;
    }

    if (localSum == (TBB_FLOAT)1233)
        a[threadIdx.x] += localSum;
}

/*
 * Pre-computed sweep sizes: STARTSIZE=100, EXPANSION=1.2, truncated to int.
 * Only sizes >= GPU_SWEEP_BLOCKSIZE are included (avoids N/BLOCKSIZE==0).
 * Covers up to ~16M elements.
 * 
 * Iterations are calculated as 1000000000 / SIZE + 2.
 */
#define DISPATCH_CASE(SIZE)                                                              \
  case SIZE:                                                                             \
    l1Kernel<SIZE, 1000000000 / SIZE + 2, GPU_SWEEP_BLOCKSIZE>                          \
        <<<numBlocks, GPU_SWEEP_BLOCKSIZE>>>(a, b, 0);                                   \
    break;

/* Sweep sizes from 512*1.2^k, filtered to >= 512, up to ~16M */
#define FOR_EACH_SWEEP_SIZE(X)                                                           \
  X(512) X(1024) X(1536) X(2048) X(2560) X(3072) X(3584) X(4096)                           \
  X(5120) X(6144) X(7680) X(9216) X(11264) X(13312) X(16384) X(19456)                      \
  X(23552) X(28160) X(33792) X(40960) X(49152) X(58880) X(70656)                           \
  X(84992) X(101888) X(122368) X(146944) X(176128) X(211456) X(253952)                     \
  X(304640) X(365568) X(438784) X(526336) X(631808) X(758272) X(909824)                    \
  X(1091584) X(1310208) X(1572352) X(1886720) X(2264064) X(2717184)                        \
  X(3260416) X(3912704) X(4695552) X(5634560) X(6761472) X(8113664)                        \
  X(9736704) X(11683840) X(14020608)

extern "C" {

/**
 * @brief Launch the template l1Kernel for agiven problem size with dispatch.
 *
 * Returns the elapsed time for the kernel execution.
 */
double gpuSweepL1Kernel(TBB_FLOAT *a, TBB_FLOAT *b,
                      size_t problemSize, int iter, int numBlocks)
{
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaDeviceSynchronize());

  double start = getTimeStamp();

  switch (problemSize) {
    FOR_EACH_SWEEP_SIZE(DISPATCH_CASE)
  default:
    fprintf(stderr,
        "Error: no template instantiation for problem size %d.\n"
        "Size must match a sweep step (STARTSIZE=100, EXPANSION=1.2, >= %d).\n",
        problemSize, GPU_SWEEP_BLOCKSIZE);
    exit(EXIT_FAILURE);
  }

  GPU_ERROR(cudaDeviceSynchronize());
  double end = getTimeStamp();

  return end - start;
}

static void gpuKernelSwitch(
    const VectorsType vec, const size_t N, const size_t iter, const int kernel)
{
  double scalar = INIT_SCALAR;
  TBB_FLOAT *a     = vec.a;
  TBB_FLOAT *b     = vec.b;
  TBB_FLOAT *c     = vec.c;
  TBB_FLOAT *d     = vec.d;

  for (int k = 0; k < iter; k++) {
    switch (kernel) {
    case INIT:
      Timings[INIT][k] = init(b, scalar, N);
      break;
    case SUM:
      Timings[SUM][k] = sum(a, N);
      break;
    case COPY:
      Timings[COPY][k] = copy(c, a, N);
      break;
    case UPDATE:
      Timings[UPDATE][k] = update(a, scalar, N);
      break;
    case TRIAD:
      Timings[TRIAD][k] = triad(a, b, c, scalar, N);
      break;
    case DAXPY:
      Timings[DAXPY][k] = daxpy(a, b, scalar, N);
      break;
    case STRIAD:
      Timings[STRIAD][k] = striad(a, b, c, d, N);
      break;
    case SDAXPY:
      Timings[SDAXPY][k] = sdaxpy(a, b, c, N);
      break;
    default:;
    }
  }
}

static void runGPUL2Sweep()
{
  printf("GPU L2 sweep mode is reserved.\n");
  exit(EXIT_SUCCESS);
}

static void runGPUSweep(VectorsType vec, const size_t N)
{
  Iterations = GPU_INCACHE_REPS;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));

  int maxThreadsPerBlock = prop.maxThreadsPerBlock;
  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
  int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;

  allocateTimer();

  for (int kernel = 0; kernel < NUMREGIONS; kernel++) {
    gpuProfilerOpenFile(kernel, NULL);
    
    THREAD_BLOCK_SIZE_SET = 1;
    THREAD_BLOCK_PER_SM_SET= 1;

    for (int tb_size = 64; tb_size <= maxThreadsPerBlock; tb_size += 64) {
      for (int tb_per_sm = 1; tb_per_sm <= maxBlocksPerSM; tb_per_sm++) {
        if (tb_size * tb_per_sm > maxThreadsPerSM) {
          continue;
        }

        THREAD_BLOCK_SIZE     = tb_size;
        THREAD_BLOCK_PER_SM   = tb_per_sm;

        gpuKernelSwitch(vec, N, Iterations, kernel);

        gpuProfilerPrintLine(N, Iterations, tb_size, tb_per_sm, kernel);
      }
    }
    gpuProfilerCloseFile();
  }

  freeTimer();
  exit(EXIT_SUCCESS);
}

static int getSMCount() {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  return prop.multiProcessorCount;
}

static int getOccupancyMaxActiveBlocks(int threadBlockSize) {
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, l1Kernel<GPU_SWEEP_BLOCKSIZE, 1000, GPU_SWEEP_BLOCKSIZE>, threadBlockSize, 0));
  return maxActiveBlocks;
}

static void runGPUL1Sweep(VectorsType vec, const size_t N)
{
  int threadBlockSize = GPU_SWEEP_BLOCKSIZE;
  Iterations = GPU_INCACHE_REPS;
  
  int smCount = getSMCount();
  int maxActiveBlocks = getOccupancyMaxActiveBlocks(threadBlockSize);
  int numThreadBlocks = smCount * 1;

  allocateTimer();

  gpuProfilerOpenFile(-1, "L1");

#define GENERATE_ARRAY_ELEMENT(SIZE) SIZE,
  const size_t sweepSizes[] = {
      FOR_EACH_SWEEP_SIZE(GENERATE_ARRAY_ELEMENT)
  };
#undef GENERATE_ARRAY_ELEMENT
  const size_t numSweepSizes = sizeof(sweepSizes) / sizeof(sweepSizes[0]);

  for (size_t i = 0; i < numSweepSizes; i++) {

    size_t problemSize = sweepSizes[i];

    size_t newN = 2 * problemSize + i * 2048;
    reinitSweepBuffers(&vec.a, &vec.b, newN);

    /* Skip sizes smaller than the thread block size */
    if (problemSize < (size_t)threadBlockSize) {
      continue;
    }

    const int iter = 1000000000 / problemSize + 2;

    /* Run the kernel GPU_INCACHE_REPS times and record timings */
    for (int k = 0; k < (int)Iterations; k++) {
      vec.a += k;
      vec.b += k;

      Timings[0][k] = gpuSweepL1Kernel(vec.a, vec.b, problemSize,
                                      iter, numThreadBlocks);
    
      vec.a -= k;
      vec.b -= k;
    }

    gpuProfilerPrintLine(problemSize, iter, threadBlockSize, numThreadBlocks, -1);
  }

  gpuProfilerCloseFile();
  freeTimer();
  exit(EXIT_SUCCESS);
}

/**
 * @brief Run GPU memory hierarchy sweeps
 *
 * Sweeps over increasing problem sizes with constant thread block size.
 */
void runGPUMemoryHierarchySweeps(VectorsType vec, const size_t N)
{
  printf(HLINE);
  printf("Running GPU sweeps\n");
  printf("Using %d repetitions per measurement.\n", GPU_INCACHE_REPS);
  printf(HLINE);

  if (GPUBenchmarkType == GPU_L2) {
    runGPUL2Sweep();
  } else if (GPUBenchmarkType == GPU_SWEEP) {
    runGPUSweep(vec, N);
  } else if (GPUBenchmarkType == GPU_L1) {
    runGPUL1Sweep(vec, N);
  }
}

} /* extern "C" */