/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>

#include "allocate.h"
#include "cli.h"
#include "constants.h"
#include "kernels.h"
#include "timing.h"

#ifdef AVX512_INTRINSICS
#include <immintrin.h>
#endif

#ifdef AVX512_INTRINSICS
#define INCREMENT i += 8
#else
#define INCREMENT i++
#endif

static void initConstants(double *, double *, double *, double *, const size_t);
static void initRandoms(double *, double *, double *, double *, const size_t);

// Adding simd clause because ICX compiler does
// not vectorise the code due to size_t dataype.
#define HARNESS(kernel)                                                                  \
  double start = getTimeStamp();                                                         \
  _Pragma("omp parallel for simd schedule(static)") for (size_t i = 0; i < N; INCREMENT) \
  {                                                                                      \
    kernel;                                                                              \
  }                                                                                      \
  double end = getTimeStamp();                                                           \
  return end - start;

void allocateArrays(double **a, double **b, double **c, double **d, const size_t N)
{
  *a = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));
  *b = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));
  *c = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));
  *d = (double *)allocate(ARRAY_ALIGNMENT, N * sizeof(double));
}

void initConstants(double *a, double *b, double *c, double *d, const size_t N)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++) {
    a[i] = INIT_A;
    b[i] = INIT_B;
    c[i] = INIT_C;
    d[i] = INIT_D;
  }
}

void initRandoms(double *a, double *b, double *c, double *d, const size_t N)
{
  printf("Using random initialization.\n");

#pragma omp parallel
  {
    unsigned int seed = time(NULL); // unique seed per thread

#pragma omp for schedule(static)
    for (size_t i = 0; i < N; i++) {
      a[i] = (double)rand_r(&seed) / RAND_MAX;
      b[i] = (double)rand_r(&seed) / RAND_MAX;
      c[i] = (double)rand_r(&seed) / RAND_MAX;
      d[i] = (double)rand_r(&seed) / RAND_MAX;
    }
  }
}

void initArrays(double *a, double *b, double *c, double *d, const size_t N)
{
  if (DataInitVariant == CONSTANT) {
    initConstants(a, b, c, d, N);
  } else if (DataInitVariant == RANDOM) {
    initRandoms(a, b, c, d, N);
  }
}

double init(double *restrict a, const double scalar, const size_t N)
{
#ifdef AVX512_INTRINSICS
  __m512d vs =
      _mm512_set_pd(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

  HARNESS(_mm512_stream_pd(&a[i], vs))
#else
  HARNESS(a[i] = scalar)
#endif
}

double sum(double *restrict a, const size_t N)
{
  double sum         = 0.0;

  const double start = getTimeStamp();
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (size_t i = 0; i < N; i++) {
    sum += a[i];
  }
  const double end = getTimeStamp();

  /* make the compiler think this makes actually sense */
  a[10] = sum;

  return end - start;
}

double update(double *restrict a, const double scalar, const size_t N)
{
#ifdef AVX512_INTRINSICS
  __m512d vs =
      _mm512_set_pd(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

  HARNESS(__m512d prod = _mm512_mul_pd(_mm512_load_pd(&a[i]), vs);
      _mm512_stream_pd(&a[i], prod);)
#else
  HARNESS(a[i] = a[i] * scalar)
#endif
}

double copy(double *restrict a, const double *restrict b, const size_t N)
{
#ifdef AVX512_INTRINSICS
  HARNESS(__m512d load = _mm512_load_pd(&b[i]); _mm512_stream_pd(&a[i], load);)
#else
  HARNESS(a[i] = b[i])
#endif
}

double triad(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double scalar,
    const size_t N)
{
#ifdef AVX512_INTRINSICS
  __m512d vs =
      _mm512_set_pd(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

  HARNESS(__m512d bvec = _mm512_load_pd(&b[i]);
      __m512d avec     = _mm512_fmadd_pd(_mm512_load_pd(&c[i]), vs, bvec);
      _mm512_stream_pd(&a[i], avec);)
#else
  HARNESS(a[i] = b[i] + (scalar * c[i]))
#endif
}

double striad(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const double *restrict d,
    const size_t N)
{
#ifdef AVX512_INTRINSICS
  HARNESS(__m512d bvec = _mm512_load_pd(&b[i]); __m512d dvec = _mm512_load_pd(&d[i]);
      __m512d avec = _mm512_fmadd_pd(_mm512_load_pd(&c[i]), dvec, bvec);
      _mm512_stream_pd(&a[i], avec);)
#else
  HARNESS(a[i] = b[i] + (d[i] * c[i]))
#endif
}

double daxpy(
    double *restrict a, const double *restrict b, const double scalar, const size_t N)
{
#ifdef AVX512_INTRINSICS
  __m512d vs =
      _mm512_set_pd(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

  HARNESS(__m512d bvec = _mm512_load_pd(&b[i]);
      __m512d avec     = _mm512_fmadd_pd(bvec, vs, _mm512_load_pd(&a[i]));
      _mm512_stream_pd(&a[i], avec);)
#else
  HARNESS(a[i] = a[i] + (scalar * b[i]))
#endif
}

double sdaxpy(double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const size_t N)
{
#ifdef AVX512_INTRINSICS
  HARNESS(__m512d bvec = _mm512_load_pd(&b[i]); __m512d cvec = _mm512_load_pd(&c[i]);
      __m512d avec = _mm512_fmadd_pd(bvec, cvec, _mm512_load_pd(&a[i]));
      _mm512_stream_pd(&a[i], avec);)
#else
  HARNESS(a[i] = a[i] + (b[i] * c[i]))
#endif
}
