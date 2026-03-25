/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#include "affinity.h"
#include <omp.h>
#endif

#include "cli.h"
#include "constants.h"
#include "kernels.h"
#include "profiler.h"
#include "util.h"

int main(const int argc, char **argv)
{
  const size_t bytesPerWord = sizeof(TBB_FLOAT);

#ifdef _OPENMP
  const size_t numThreads = omp_get_max_threads();
#else
  const size_t numThreads = 1;
#endif

  // Round up N so each thread gets an 8-aligned chunk
  const size_t alignment        = sizeof(TBB_FLOAT);
  const size_t perThread        = (N + numThreads - 1) / numThreads; // Ceiling division
  const size_t alignedPerThread = (perThread + alignment - 1) & ~(alignment - 1);
  N                             = alignedPerThread * numThreads;

  profilerInit();
  parseArguments(argc, argv);
  allocateTimer();

  printf("\n");
  printf(BANNER);
  printf(HLINE);
  printf("Total allocated datasize: %8.2f MB\n",
      NUMVECTORS * (double)(bytesPerWord * N) * MILLIONTH);

#ifdef _OPENMP
  printf("OpenMP enabled, running with %zu threads\n", numThreads);

#ifdef VERBOSE_AFFINITY
#pragma omp parallel
  {
    int i = omp_get_thread_num();
#pragma omp critical
    {
      printf("Thread %d running on processor %d\n", i, affinity_getProcessorId());
      affinity_getmask();
    }
  }
#endif
#else
  Sequential = true;
#endif

  VectorsType vec;
  allocateArrays(&vec.a, &vec.b, &vec.c, &vec.d, N);
  initArrays(vec.a, vec.b, vec.c, vec.d, N);

  runBenchmarks(vec, N);

  profilerPrint(N);
  freeTimer();

  return EXIT_SUCCESS;
}
