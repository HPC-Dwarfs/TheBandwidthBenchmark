/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef CLI_H
#define CLI_H

#include <stdbool.h>
#include <stddef.h>

typedef enum { WS = 0, TP, SQ, NUMTYPES } ModeType;
typedef enum { CONSTANT = 0, RANDOM } InitType;
typedef enum { VEC0 = 0, VEC2, VEC4} VectorizedDataTransferType;

#define HELPTEXT                                                                         \
  "Usage: bwBench [options]\n\n"                                                         \
  "Options:\n"                                                                           \
  "  -h              Show this help text\n"                                              \
  "  -m <type>       Benchmark type, can be ws (default), tp, or seq.\n"                 \
  "  -s <long int>   Size in GB for allocated vectors\n"                                 \
  "  -n <long int>   Number of iterations\n"                                             \
  "  -i <type>       Data initialization type, can be constant, or random\n"             \
  "  -d <int>        (If GPU enabled) GPU ID to execute on\n"

extern int BenchmarkType;
extern bool Sequential;
extern size_t N;
extern size_t Iterations;
extern int DataInitVariant;

#if defined(_NVCC) || defined(_HIP)
extern int CUDA_DEVICE;
extern int THREAD_BLOCK_SIZE;
extern int THREAD_BLOCK_SIZE_SET;
extern int THREAD_BLOCK_PER_SM;
extern int THREAD_BLOCK_PER_SM_SET;
extern int VEC_VARIANT;
#endif

extern void parseArguments(int, char **);

#endif /*CLI_H*/
