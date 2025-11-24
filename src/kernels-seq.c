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

double initSeq(double *restrict a, const double scalar, const size_t N, const size_t iter)
{
  HARNESS(a[i] = scalar)
}

double updateSeq(
    double *restrict a, const double scalar, const size_t N, const size_t iter)
{
  HARNESS(a[i] = a[i] * scalar)
}

double copySeq(
    double *restrict a, const double *restrict b, const size_t N, const size_t iter)
{
  HARNESS(a[i] = b[i])
}

double triadSeq(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = b[i] + (scalar * c[i]))
}

double striadSeq(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double *restrict d,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = b[i] + (d[i] * c[i]))
}

double daxpySeq(double *restrict a,
    const double *restrict b,
    const double scalar,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = a[i] + (scalar * b[i]))
}

double sdaxpySeq(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const size_t N,
    const size_t iter)
{
  HARNESS(a[i] = a[i] + (b[i] * c[i]))
}

double sumSeq(double *restrict a, const size_t N, const size_t iter)
{
  double sum         = 0.0;

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
