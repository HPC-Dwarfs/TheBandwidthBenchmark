/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef CLI_H
#define CLI_H

#include <stdbool.h>
#include <stddef.h>

#include "util.h"

#ifndef _GPU
typedef enum { WS = 0, TP, SQ, NUMTYPES } ModeType;
#else
typedef enum { GPU_WS = 0, GPU_L1, GPU_L2, GPU_SWEEP, GPU_NUMTYPES } GPUModeType;
#endif

typedef enum { CONSTANT = 0, RANDOM } InitType;
typedef enum { VEC0 = 0, VEC2, VEC4} VectorizedDataTransferType;

#define HELPTEXT                                                                         \
  "Usage: bwBench [options]\n\n"                                                         \
  "Options:\n"                                                                           \
  "  -h              Show this help text\n"                                              \
  "  -m <type>       Benchmark type: ws (default), tp, seq (CPU); l1, l2, sweep (GPU)\n"\
  "  -s <long int>   Size in GB for allocated vectors\n"                                 \
  "  -n <long int>   Number of iterations\n"                                             \
  "  -i <type>       Data initialization type, can be constant, or random\n"             \
  "  -d <int>        (If GPU enabled) GPU ID to execute on\n"

extern int BenchmarkType;
extern bool Sequential;
extern size_t N;
extern size_t Iterations;
extern int DataInitVariant;

#ifdef _GPU
extern int GPUBenchmarkType;
extern int CUDA_DEVICE;
extern int THREAD_BLOCK_SIZE;
extern int THREAD_BLOCK_SIZE_SET;
extern int THREAD_BLOCK_PER_SM;
extern int THREAD_BLOCK_PER_SM_SET;
extern int VEC_VARIANT;
#endif

extern void parseArguments(int, char **);

#endif /*CLI_H*/
