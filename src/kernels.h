/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef KERNELS_H_
#define KERNELS_H_
#include <stdlib.h>
#include <time.h>

extern void allocateArrays(double **a, double **b, double **c, double **d, size_t N);
extern void initArrays(double *a, double *b, double *c, double *d, size_t N);
extern double init(double *a, double scalar, size_t N);
extern double sum(double *a, size_t N);
extern double update(double *a, double scalar, size_t N);
extern double copy(double *a, const double *b, size_t N);
extern double triad(double *a, const double *b, const double *c, double scalar, size_t N);
extern double striad(
    double *a, const double *b, const double *c, const double *d, size_t N);
extern double daxpy(double *a, const double *b, double scalar, size_t N);
extern double sdaxpy(double *a, const double *b, const double *c, size_t N);

#ifndef _NVCC
extern double initSeq(double *a, double scalar, size_t N, size_t iter);
extern double updateSeq(double *a, double scalar, size_t N, size_t iter);
extern double sumSeq(double *a, size_t N, size_t iter);
extern double copySeq(double *a, const double *b, size_t N, size_t iter);
extern double triadSeq(
    double *a, const double *b, const double *c, double scalar, size_t N, size_t iter);
extern double striadSeq(
    double *a, const double *b, const double *c, const double *d, size_t N, size_t iter);
extern double daxpySeq(double *a, const double *b, double scalar, size_t N, size_t iter);
extern double sdaxpySeq(
    double *a, const double *b, const double *c, size_t N, size_t iter);

extern double initTp(double *a, double scalar, size_t N, size_t iter);
extern double updateTp(const double *a, double scalar, size_t N, size_t iter);
extern double sumTp(const double *a, size_t N, size_t iter);
extern double copyTp(double *a, const double *b, size_t N, size_t iter);
extern double triadTp(
    double *a, const double *b, const double *c, double scalar, size_t N, size_t iter);
extern double striadTp(
    double *a, const double *b, const double *c, const double *d, size_t N, size_t iter);
extern double daxpyTp(
    const double *a, const double *b, double scalar, size_t N, size_t iter);
extern double sdaxpyTp(
    const double *a, const double *b, const double *c, size_t N, size_t iter);
#endif
#endif
