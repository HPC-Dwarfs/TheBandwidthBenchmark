/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>

#include "allocate.h"
#include "kernels.h"
#include "timing.h"

#define HARNESS(kernel)                                                                  \
  double S, E;                                                                           \
  _Pragma("omp parallel")                                                                \
  {                                                                                      \
    double *al              = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));   \
    _Pragma("omp single") S = getTimeStamp();                                            \
    for (size_t j = 0; j < iter; j++) {                                                  \
      _Pragma("omp simd") for (size_t i = 0; i < N; i++)                                 \
      {                                                                                  \
        kernel;                                                                          \
      }                                                                                  \
      if (al[N - 1] < 0.0)                                                               \
        printf("Ai = %f\n", al[N - 1]);                                                  \
    }                                                                                    \
    _Pragma("omp barrier") _Pragma("omp single") E = getTimeStamp();                     \
    free(al);                                                                            \
  }                                                                                      \
  return E - S;

double initTp(double *restrict a, const double scalar, const size_t N, const size_t iter)
{
  HARNESS(al[i] = scalar)
}

double updateTp(
    const double *restrict a, const double scalar, const size_t N, const size_t iter)
{
  HARNESS(al[i] = a[i] * scalar)
}

double copyTp(
    double *restrict a, const double *restrict b, const size_t N, const size_t iter)
{
  HARNESS(al[i] = b[i])
}

double triadTp(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(al[i] = b[i] + scalar * c[i])
}

double striadTp(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double *restrict d,
    const size_t N,
    const size_t iter)
{
  HARNESS(al[i] = b[i] + d[i] * c[i])
}

double daxpyTp(const double *restrict a,
    const double *restrict b,
    const double scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(al[i] = a[i] + scalar * b[i])
}

double sdaxpyTp(const double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const size_t N,
    const size_t iter)
{
  HARNESS(al[i] = a[i] + b[i] * c[i])
}

double sumTp(const double *restrict a, const size_t N, const size_t iter)
{
  double start;
  double end;

  _Pragma("omp parallel")
  {
    double *al = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));
    _Pragma("omp simd") for (size_t i = 0; i < N; i++)
    {
      al[i] = a[i];
    }
    double sum                  = 0.0;

    _Pragma("omp single") start = getTimeStamp();
    for (size_t j = 0; j < iter; j++) {
      _Pragma("omp simd") for (size_t i = 0; i < N; i++)
      {
        sum += al[i];
      }
      al[N / 2] += sum;
    }
    _Pragma("omp single") end = getTimeStamp();

    free(al);
  }

  /* make the compiler think this makes actually sense */

  return end - start;
}
