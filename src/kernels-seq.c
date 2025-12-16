/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>

#include "kernels.h"
#include "timing.h"

#define HARNESS(kernel)                                                                  \
  const double S = getTimeStamp();                                                       \
  for (size_t j = 0; j < iter; j++) {                                                    \
    for (size_t i = 0; i < N; i++) {                                                     \
      kernel;                                                                            \
    }                                                                                    \
    if (a[N - 1] < 0.0) {                                                                \
      printf("Ai = %f\n", a[N - 1]);                                                     \
      exit(1);                                                                           \
    }                                                                                    \
  }                                                                                      \
  const double E = getTimeStamp();                                                       \
  return E - S;

double initSeq(TBB_FLOAT *restrict a, const TBB_FLOAT scalar, const size_t N, const size_t iter)
{
  HARNESS(a[i] = scalar)
}

double updateSeq(
    TBB_FLOAT *restrict a, const TBB_FLOAT scalar, const size_t N, const size_t iter)
{
  HARNESS(a[i] = a[i] * scalar)
}

double copySeq(
    TBB_FLOAT *restrict a, const TBB_FLOAT *restrict b, const size_t N, const size_t iter)
{
  HARNESS(a[i] = b[i])
}

double triadSeq(TBB_FLOAT *restrict a,
    const TBB_FLOAT *restrict b,
    const TBB_FLOAT *restrict c,
    const TBB_FLOAT scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = b[i] + (scalar * c[i]))
}

double striadSeq(TBB_FLOAT *restrict a,
    const TBB_FLOAT *restrict b,
    const TBB_FLOAT *restrict c,
    const TBB_FLOAT *restrict d,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = b[i] + (d[i] * c[i]))
}

double daxpySeq(TBB_FLOAT *restrict a,
    const TBB_FLOAT *restrict b,
    const TBB_FLOAT scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = a[i] + (scalar * b[i]))
}

double sdaxpySeq(TBB_FLOAT *restrict a,
    const TBB_FLOAT *restrict b,
    const TBB_FLOAT *restrict c,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = a[i] + (b[i] * c[i]))
}

double sumSeq(TBB_FLOAT *restrict a, const size_t N, const size_t iter)
{
  TBB_FLOAT sum         = 0.0;

  const double start = getTimeStamp();
  for (size_t j = 0; j < iter; j++) {
    for (size_t i = 0; i < N; i++) {
      sum += a[i];
    }

    a[10] = sum;
  }
  const double end = getTimeStamp();

  /* make the compiler think this makes actually sense */
  a[10] = sum;

  return end - start;
}
