/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __PROFILER_H_
#define __PROFILER_H_
#include <stddef.h>

#ifdef _OPENMP
#include "likwid-marker.h"

#define PROFILE(tag, call)                                                               \
  _Pragma("omp parallel default(none)")                                                  \
  {                                                                                      \
    LIKWID_MARKER_START(#tag);                                                           \
  }                                                                                      \
  Timings[tag][k] = call;                                                                \
  _Pragma("omp parallel default(none)")                                                  \
  {                                                                                      \
    LIKWID_MARKER_STOP(#tag);                                                            \
  }
#else
#define PROFILE(tag, call) Timings[tag][k] = call;
#endif

typedef enum {
  INIT = 0,
  SUM,
  COPY,
  UPDATE,
  TRIAD,
  DAXPY,
  STRIAD,
  SDAXPY,
  NUMREGIONS
} RegionType;

extern double **Timings;
extern void allocateTimer();
extern void freeTimer();
extern void profilerInit();
extern void profilerPrint(size_t N);
extern void profilerOpenFile(int region);
extern void profilerCloseFile(void);
extern void profilerPrintLine(size_t N, size_t iter, int kernel);

#endif // __PROFILER_H
