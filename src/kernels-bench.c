/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>

#include "cli.h"
#include "constants.h"
#include "kernels.h"
#include "profiler.h"
#include "util.h"

static void check(VectorsType vec, size_t N, size_t iter);
static size_t findIter(VectorsType vec, size_t iter, size_t problemSize);
static void kernelSwitch(VectorsType vec, size_t N, size_t iter, int kernel);
static void runMemoryHierarchySweeps(VectorsType vec, size_t N);

#define RUN_KERNEL(operation, label, ...)                                                \
  for (int k = 0; k < Iterations; k++) {                                                 \
    Timings[label][k] = operation(__VA_ARGS__);                                          \
  }

void runBenchmarks(VectorsType vec, const size_t N)
{
  if (BenchmarkType == TP || BenchmarkType == SQ) {
    runMemoryHierarchySweeps(vec, N);
  }

  const double scalar = INIT_SCALAR;
  TBB_FLOAT *a           = vec.a;
  TBB_FLOAT *b           = vec.b;
  TBB_FLOAT *c           = vec.c;
  TBB_FLOAT *d           = vec.d;
  printf("Doing %zu repetitions per kernel\n", Iterations);

  for (int k = 0; k < Iterations; k++) {
    PROFILE(INIT, init(b, scalar, N));
    const double tmp = a[10];
    PROFILE(SUM, sum(a, N));
    a[10] = tmp;
    PROFILE(COPY, copy(c, a, N));
    PROFILE(UPDATE, update(a, scalar, N));
    PROFILE(TRIAD, triad(a, b, c, scalar, N));
    PROFILE(DAXPY, daxpy(a, b, scalar, N));
    PROFILE(STRIAD, striad(a, b, c, d, N));
    PROFILE(SDAXPY, sdaxpy(a, b, c, N));
  }

  check(vec, N, Iterations);
}

static void check(const VectorsType vec, const size_t n, const size_t ITERS)
{
  if (DataInitVariant == RANDOM) {
    return;
  }

  /* reproduce initialization */
  TBB_FLOAT aj           = INIT_A;
  TBB_FLOAT bj           = INIT_B;
  TBB_FLOAT cj           = INIT_C;
  TBB_FLOAT dj           = INIT_D;
  const TBB_FLOAT scalar = INIT_SCALAR;

  /* now execute timing loop */
  for (int k = 0; k < ITERS; k++) {
    bj = scalar;
    cj = aj;
    aj = aj * scalar;
    aj = bj + (scalar * cj);
    aj = aj + (scalar * bj);
    aj = bj + (cj * dj);
    aj = aj + (bj * cj);
  }

  aj          = aj * (double)(n);
  bj          = bj * (double)(n);
  cj          = cj * (double)(n);
  dj          = dj * (double)(n);

  TBB_FLOAT asum = 0.0;
  TBB_FLOAT bsum = 0.0;
  TBB_FLOAT csum = 0.0;
  TBB_FLOAT dsum = 0.0;
  TBB_FLOAT *a   = vec.a;
  TBB_FLOAT *b   = vec.b;
  TBB_FLOAT *c   = vec.c;
  TBB_FLOAT *d   = vec.d;

  for (size_t i = 0; i < n; i++) {
    asum += a[i];
    bsum += b[i];
    csum += c[i];
    dsum += d[i];
  }

#ifdef VERBOSE
  printf("Results Comparison: \n");
  printf("        Expected  : %f %f %f \n", aj, bj, cj);
  printf("        Observed  : %f %f %f \n", asum, bsum, csum);
#endif

  double epsilon = CHECK_MAX_EPSILON;

  if (ABS(aj - asum) / asum > epsilon) {
    printf("Failed Validation on array a[]\n");
    printf("        Expected  : %f \n", aj);
    printf("        Observed  : %f \n", asum);
  } else if (ABS(bj - bsum) / bsum > epsilon) {
    printf("Failed Validation on array b[]\n");
    printf("        Expected  : %f \n", bj);
    printf("        Observed  : %f \n", bsum);
  } else if (ABS(cj - csum) / csum > epsilon) {
    printf("Failed Validation on array c[]\n");
    printf("        Expected  : %f \n", cj);
    printf("        Observed  : %f \n", csum);
  } else if (ABS(dj - dsum) / dsum > epsilon) {
    printf("Failed Validation on array d[]\n");
    printf("        Expected  : %f \n", dj);
    printf("        Observed  : %f \n", dsum);
  } else {
    printf("Solution Validates\n");
  }
}

static void kernelSwitch(
    const VectorsType vec, const size_t N, const size_t iter, const int kernel)
{
  double scalar = INIT_SCALAR;
  TBB_FLOAT *a     = vec.a;
  TBB_FLOAT *b     = vec.b;
  TBB_FLOAT *c     = vec.c;
  TBB_FLOAT *d     = vec.d;

  switch (kernel) {
  case INIT:
    if (Sequential) {
      RUN_KERNEL(initSeq, INIT, a, scalar, N, iter);
    } else {
      RUN_KERNEL(initTp, INIT, a, scalar, N, iter);
    }
    break;

  case SUM:
    if (Sequential) {
      RUN_KERNEL(sumSeq, SUM, a, N, iter);
    } else {
      RUN_KERNEL(sumTp, SUM, a, N, iter);
    }
    break;

  case COPY:
    if (Sequential) {
      RUN_KERNEL(copySeq, COPY, a, b, N, iter);
    } else {
      RUN_KERNEL(copyTp, COPY, a, b, N, iter);
    }
    break;

  case UPDATE:
    if (Sequential) {
      RUN_KERNEL(updateSeq, UPDATE, a, scalar, N, iter);
    } else {
      RUN_KERNEL(updateTp, UPDATE, a, scalar, N, iter);
    }
    break;

  case TRIAD:
    if (Sequential) {
      RUN_KERNEL(triadSeq, TRIAD, a, b, c, scalar, N, iter);
    } else {
      RUN_KERNEL(triadTp, TRIAD, a, b, c, scalar, N, iter);
    }
    break;

  case DAXPY:
    if (Sequential) {
      RUN_KERNEL(daxpySeq, DAXPY, a, b, scalar, N, iter);
    } else {
      RUN_KERNEL(daxpyTp, DAXPY, a, b, scalar, N, iter);
    }
    break;

  case STRIAD:
    if (Sequential) {
      RUN_KERNEL(striadSeq, STRIAD, a, b, c, d, N, iter);
    } else {
      RUN_KERNEL(striadTp, STRIAD, a, b, c, d, N, iter);
    }
    break;

  case SDAXPY:
    if (Sequential) {
      RUN_KERNEL(sdaxpySeq, SDAXPY, a, b, c, N, iter);
    } else {
      RUN_KERNEL(sdaxpyTp, SDAXPY, a, b, c, N, iter);
    }
    break;
  default:;
  }
}

static size_t findIter(const VectorsType vec, size_t iter, const size_t problemSize)
{
  // Target runtime: 0.3 seconds for reliable measurements
  const double targetTime   = 0.3;
  const double minTime      = 0.1;
  const double fallbackTime = 0.005;
  const double safetyFactor = 0.9;
  TBB_FLOAT *a                 = vec.a;
  TBB_FLOAT *b                 = vec.b;
  TBB_FLOAT *c                 = vec.c;
  TBB_FLOAT *d                 = vec.d;

  while (1) {
    double newtime = striadSeq(a, b, c, d, problemSize, iter);
    // printf("newtime: %d %e\n", (int)iter, newtime);

    if (newtime >= minTime && newtime <= targetTime) {
      // Found acceptable iteration count
      break;
    }

    if (newtime < minTime) {
      if (newtime == 0.0) {
        newtime = fallbackTime;
      }
      // Too fast, increase iterations proportionally
      const double factor = targetTime / newtime;
      iter                = iter * (size_t)(factor * safetyFactor);
      if (iter < 2) {
        iter = 2;
      }
    } else {
      // Too slow, we're done
      break;
    }
  }

  return iter;
}

static void runMemoryHierarchySweeps(VectorsType vec, const size_t N)
{
  Iterations = INCACHE_REPS;
  printf("Running memory hierarchy sweeps\n");
  printf("Using %zu repetitions per measurement.\n", Iterations);

  for (int kernel = 0; kernel < NUMREGIONS; kernel++) {
    size_t problemSize = STARTSIZE;
    profilerOpenFile(kernel);

    while (problemSize < N) {

      const size_t iter = findIter(vec, 2, problemSize);
      kernelSwitch(vec, problemSize, iter, kernel);
      profilerPrintLine(problemSize, iter, kernel);
      problemSize = problemSize * EXPANSION;
    }

    profilerCloseFile();
  }
  exit(EXIT_SUCCESS);
}
