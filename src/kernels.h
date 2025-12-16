/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef KERNELS_H_
#define KERNELS_H_
#include <stdlib.h>
#include <time.h>
#include "util.h"

extern void allocateArrays(
    TBB_FLOAT **a, TBB_FLOAT **b, TBB_FLOAT **c, TBB_FLOAT **d, size_t N);
extern void initArrays(TBB_FLOAT *a, TBB_FLOAT *b, TBB_FLOAT *c, TBB_FLOAT *d, size_t N);
extern double init(TBB_FLOAT *a, TBB_FLOAT scalar, size_t N);
extern double sum(TBB_FLOAT *a, size_t N);
extern double update(TBB_FLOAT *a, TBB_FLOAT scalar, size_t N);
extern double copy(TBB_FLOAT *a, const TBB_FLOAT *b, size_t N);
extern double triad(
    TBB_FLOAT *a, const TBB_FLOAT *b, const TBB_FLOAT *c, TBB_FLOAT scalar, size_t N);
extern double striad(
    TBB_FLOAT *a, const TBB_FLOAT *b, const TBB_FLOAT *c, const TBB_FLOAT *d, size_t N);
extern double daxpy(TBB_FLOAT *a, const TBB_FLOAT *b, TBB_FLOAT scalar, size_t N);
extern double sdaxpy(TBB_FLOAT *a, const TBB_FLOAT *b, const TBB_FLOAT *c, size_t N);

#ifndef _NVCC
extern double initSeq(TBB_FLOAT *a, TBB_FLOAT scalar, size_t N, size_t iter);
extern double updateSeq(TBB_FLOAT *a, TBB_FLOAT scalar, size_t N, size_t iter);
extern double sumSeq(TBB_FLOAT *a, size_t N, size_t iter);
extern double copySeq(TBB_FLOAT *a, const TBB_FLOAT *b, size_t N, size_t iter);
extern double triadSeq(TBB_FLOAT *a,
    const TBB_FLOAT *b,
    const TBB_FLOAT *c,
    TBB_FLOAT scalar,
    size_t N,
    size_t iter);
extern double striadSeq(TBB_FLOAT *a,
    const TBB_FLOAT *b,
    const TBB_FLOAT *c,
    const TBB_FLOAT *d,
    size_t N,
    size_t iter);
extern double daxpySeq(
    TBB_FLOAT *a, const TBB_FLOAT *b, TBB_FLOAT scalar, size_t N, size_t iter);
extern double sdaxpySeq(
    TBB_FLOAT *a, const TBB_FLOAT *b, const TBB_FLOAT *c, size_t N, size_t iter);

extern double initTp(TBB_FLOAT *a, TBB_FLOAT scalar, size_t N, size_t iter);
extern double updateTp(const TBB_FLOAT *a, TBB_FLOAT scalar, size_t N, size_t iter);
extern double sumTp(const TBB_FLOAT *a, size_t N, size_t iter);
extern double copyTp(TBB_FLOAT *a, const TBB_FLOAT *b, size_t N, size_t iter);
extern double triadTp(TBB_FLOAT *a,
    const TBB_FLOAT *b,
    const TBB_FLOAT *c,
    TBB_FLOAT scalar,
    size_t N,
    size_t iter);
extern double striadTp(TBB_FLOAT *a,
    const TBB_FLOAT *b,
    const TBB_FLOAT *c,
    const TBB_FLOAT *d,
    size_t N,
    size_t iter);
extern double daxpyTp(
    const TBB_FLOAT *a, const TBB_FLOAT *b, TBB_FLOAT scalar, size_t N, size_t iter);
extern double sdaxpyTp(
    const TBB_FLOAT *a, const TBB_FLOAT *b, const TBB_FLOAT *c, size_t N, size_t iter);
#endif
#endif
