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

extern "C" {

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

void runGPUSweep(VectorsType vec, const size_t N)
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

} /* extern "C" */
