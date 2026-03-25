#include "portability.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "cli.h"
#include "constants.h"
#include "kernels.h"
#include "profiler.h"
#include "timing.h"
#include "util.h"
}

#define L2_N_ELEMENTS 64
#define L2_BLOCK_COUNT 200000

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

__global__ void l2Kernel(
    TBB_FLOAT *__restrict__ a, const TBB_FLOAT *__restrict__ b, int blockRun)
{
  TBB_FLOAT localSum = (TBB_FLOAT)0;

  for (int i = 0; i < L2_N_ELEMENTS / 2; i++) {
    int idx = (blockDim.x * blockRun * i + (blockIdx.x % blockRun) * blockDim.x) * 2 +
              threadIdx.x;
    localSum += b[idx] * b[idx + blockDim.x];
  }

  localSum *= (TBB_FLOAT)1.3;
  if (threadIdx.x > 1233 || localSum == (TBB_FLOAT)23.12)
    a[threadIdx.x] += localSum;
}

extern "C" {

static double gpuSweepL2Kernel(TBB_FLOAT *a, TBB_FLOAT *b, int blockRun)
{
  GPU_ERROR(cudaSetDevice(CUDA_DEVICE));
  GPU_ERROR(cudaDeviceSynchronize());

  double start = getTimeStamp();

  l2Kernel<<<L2_BLOCK_COUNT, GPU_SWEEP_BLOCKSIZE>>>(a, b, blockRun);

  GPU_ERROR(cudaDeviceSynchronize());
  double end = getTimeStamp();

  return end - start;
}

void runGPUL2Sweep(VectorsType vec, const size_t N)
{
  Iterations = GPU_INCACHE_REPS;

  allocateTimer();
  gpuProfilerOpenFile(-2, "L2");

  for (int blockRun = 3; blockRun < 10000; blockRun += (int)fmax(1.0, blockRun * 0.1)) {

    size_t bufferCount = (size_t)blockRun * GPU_SWEEP_BLOCKSIZE * L2_N_ELEMENTS;

    reinitSweepBuffers(&vec.a, &vec.b, bufferCount);

    for (int k = 0; k < (int)Iterations; k++) {

      Timings[0][k] = gpuSweepL2Kernel(vec.a, vec.b, blockRun);
    }

    /* N=bufferCount, iter=blockCount, threadBlockSize=blockSize, numThreadBlocks=blockRun */
    gpuProfilerPrintLine(bufferCount, L2_BLOCK_COUNT, GPU_SWEEP_BLOCKSIZE, blockRun, -2);
  }

  gpuProfilerCloseFile();
  freeTimer();
  exit(EXIT_SUCCESS);
}


} /* extern "C" */
