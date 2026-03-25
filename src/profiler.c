/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli.h"
#include "constants.h"
#include "likwid-marker.h"
#include "profiler.h"
#include "util.h"

typedef struct {
  char *label;
  size_t words;
  size_t flops;
} WorkType;

// double timings[NUMREGIONS][ITERS];
double **Timings;
FILE *ProfilerFile                  = NULL;
char *DataDirectory                 = "dat\0";

static WorkType Regions[NUMREGIONS] = {
  { "Init",   1, 0 },
  { "Sum",    1, 1 },
  { "Copy",   2, 0 },
  { "Update", 2, 1 },
  { "Triad",  3, 2 },
  { "Daxpy",  3, 2 },
  { "STriad", 4, 2 },
  { "SDaxpy", 4, 2 }
};

void profilerInit(void)
{
  LIKWID_MARKER_INIT;
  _Pragma("omp parallel default(none)")
  {
    LIKWID_MARKER_REGISTER("INIT");
    LIKWID_MARKER_REGISTER("SUM");
    LIKWID_MARKER_REGISTER("COPY");
    LIKWID_MARKER_REGISTER("UPDATE");
    LIKWID_MARKER_REGISTER("TRIAD");
    LIKWID_MARKER_REGISTER("DAXPY");
    LIKWID_MARKER_REGISTER("STRIAD");
    LIKWID_MARKER_REGISTER("SDAXPY");
  }
}

static void computeStats(double *avgtime, double *maxtime, double *mintime, const int j)
{
  *avgtime = 0;
  *maxtime = 0;
  *mintime = FLT_MAX;

  for (int k = 1; k < Iterations; k++) {
    *avgtime += Timings[j][k];
    *mintime = MIN(*mintime, Timings[j][k]);
    *maxtime = MAX(*maxtime, Timings[j][k]);
  }

  *avgtime /= (double)(Iterations - 1);
}

void allocateTimer()
{
  Timings = (double **)malloc(NUMREGIONS * sizeof(double *));
  for (int i = 0; i < NUMREGIONS; i++) {
    Timings[i] = malloc(Iterations * sizeof(double));
  }
}

void freeTimer()
{
  for (int i = 0; i < NUMREGIONS; i++) {
    free(Timings[i]);
  }
  free((void *)Timings);
}

void profilerOpenFile(const int region)
{
  char filename[MAXSTRLEN];
  sprintf(filename, "%s/%s.dat", DataDirectory, Regions[region].label);
  ProfilerFile = fopen(filename, "w");
  if (Regions[region].flops == 0) {
    FPRINTF(ProfilerFile,
        "# %s: %lu words, no flops\n",
        Regions[region].label,
        Regions[region].words);
    FPRINTF(ProfilerFile,
        "# N  Bytes(MB)  Rate(GB/s)  Avg time(s)  Min time(s)  Max time(s)\n");
  } else {
    FPRINTF(ProfilerFile,
        "# %s: %lu words, %lu flops\n",
        Regions[region].label,
        Regions[region].words,
        Regions[region].words);
    FPRINTF(ProfilerFile,
        "# N  Bytes(MB)  Rate(GB/s)  Rate(GFlop/s)  Avg time(s)  Min time(s)  "
        "Max time(s)\n");
  }

  printf("Running kernel %s\n", Regions[region].label);
}

void profilerCloseFile(void)
{
  if (fclose(ProfilerFile) != 0) {
    perror("Error closing profiler file");
  }
}

void profilerPrintLine(const size_t N, const size_t iter, const int kernel)
{
  size_t bytesPerWord = sizeof(TBB_FLOAT);
  double avgtime;
  double maxtime;
  double mintime;
  size_t numThreads = 1;

#ifdef _OPENMP
  if (!Sequential) {
    numThreads = omp_get_max_threads();
  }
#else
#endif

  computeStats(&avgtime, &maxtime, &mintime, kernel);
  double bytes =
      (double)Regions[kernel].words * sizeof(TBB_FLOAT) * (double)(N * numThreads);
  double flops = (double)Regions[kernel].flops * (double)(N * iter * numThreads);
  //double bytes = (double)Regions[j].words * sizeof(double) * N * numThreads;
  //double flops = (double)Regions[j].flops * N * iter * numThreads;

  // N  Bytes(MB)  Rate(GB/s)  Rate(MFlop/s)  Avg time(s)  Min time(s)  Max
  // time(s)
  if (flops > 0) {
    FPRINTF(ProfilerFile,
        "%lu %11.5f %11.2f %11.2f %11.4f  %11.4f  %11.4f\n",
        N,
        MILLIONTH * bytes,
        BILLIONTH * bytes * iter / mintime,
        BILLIONTH * flops / mintime,
        avgtime,
        mintime,
        maxtime);
  }
  // N  Bytes(MB)  Rate(GB/s)  Avg time(s)  Min time(s)  Max time(s)
  else {
    FPRINTF(ProfilerFile,
        "%lu %11.5f %11.2f %11.4f  %11.4f  %11.4f\n",
        N,
        MILLIONTH * bytes,
        BILLIONTH * bytes * iter / mintime,
        avgtime,
        mintime,
        maxtime);
  }
}

void profilerPrint(const size_t N)
{
  double avgtime;
  double maxtime;
  double mintime;

#ifdef VERBOSE_DATASIZE
  size_t bytesPerWord = sizeof(TBB_FLOAT);
  printf(HLINE);
  printf("Dataset sizes\n");
  for (int i = 0; i < NUMREGIONS; i++) {
    printf("%s: %8.2f MB\n",
        _regions[i].label,
        _regions[i].words * bytesPerWord * N * 1.0E-06);
  }
#endif

  printf(HLINE);
  printf("Function      Rate(GB/s)  Rate(GFlop/s)  Avg time     Min time     "
         "Max time\n");

  for (int j = 0; j < NUMREGIONS; j++) {
    computeStats(&avgtime, &maxtime, &mintime, j);
    const double bytes = (double)Regions[j].words * sizeof(TBB_FLOAT) * (double)N;
    const double flops = (double)Regions[j].flops * (double)N;

    if (flops > 0) {
      printf("%-12s%11.2f %11.2f %11.4f  %11.4f  %11.4f\n",
          Regions[j].label,
          BILLIONTH * bytes / mintime,
          BILLIONTH * flops / mintime,
          avgtime,
          mintime,
          maxtime);
    } else {
      printf("%-12s%11.2f      -      %11.4f  %11.4f  %11.4f\n",
          Regions[j].label,
          BILLIONTH * bytes / mintime,
          avgtime,
          mintime,
          maxtime);
    }
  }
  printf(HLINE);

  LIKWID_MARKER_CLOSE;
}

#if defined(_NVCC) || defined(_HIP)
void gpuProfilerOpenFile(int region, const char *label)
{
  char filename[MAXSTRLEN];
  const char *name = (region >= 0) ? Regions[region].label : label;
  int hasFlops = (region >= 0) && (Regions[region].flops > 0);

  sprintf(filename, "%s/%s.dat", DataDirectory, name);
  ProfilerFile = fopen(filename, "w");
  if (ProfilerFile == NULL) {
    perror("Error opening GPU profiler file");
    exit(EXIT_FAILURE);
  }

  FPRINTF(ProfilerFile, "# GPU Sweep: %s\n", name);
  if (hasFlops) {
    FPRINTF(ProfilerFile,
        "# N  DatasetSize(MB)  Rate(GB/s)  Rate(GFlop/s)  ThreadBlockSize  NumThreadBlocks"
        "  Avg_time(s)  Min_time(s)  Max_time(s)\n");
  } else {
    FPRINTF(ProfilerFile,
        "# N  DatasetSize(MB)  Rate(GB/s)  ThreadBlockSize  NumThreadBlocks"
        "  Avg_time(s)  Min_time(s)  Max_time(s)\n");
  }

  printf("Measuring GPU kernel %s\n", name);
}

void gpuProfilerPrintLine(const size_t N, const int iter,
    const int threadBlockSize, const int numThreadBlocks, int region)
{
  double avgtime, maxtime, mintime;
  computeStats(&avgtime, &maxtime, &mintime, region >= 0 ? region : 0);

  double dataset, rate;

  if (region >= 0) {
    dataset = (double)Regions[region].words * sizeof(TBB_FLOAT) * (double)N;
    rate = dataset / mintime;
  } else if (region == -2) {
    /* L2 mode: N=bufferCount, iter=blockCount, numThreadBlocks=blockRun */
    dataset = (double)N * sizeof(TBB_FLOAT);
    rate = ((double)N / numThreadBlocks) * sizeof(TBB_FLOAT) * (double)iter / mintime;
  } else {
    /* L1 mode */
    dataset = 2.0 * sizeof(TBB_FLOAT) * (double)N;
    rate = 2.0 * sizeof(TBB_FLOAT) * (double)N * (double)iter * (double)numThreadBlocks / mintime;
  }

  if (region >= 0 && Regions[region].flops > 0) {
    double flops = (double)Regions[region].flops * (double)N;
    FPRINTF(ProfilerFile,
        "%lu %11.3f %11.2f %11.2f %15d %15d %12.6f  %12.6f  %12.6f\n",
        N, MILLIONTH * dataset, BILLIONTH * rate,
        BILLIONTH * flops / mintime,
        threadBlockSize, numThreadBlocks, avgtime, mintime, maxtime);
  } else {
    FPRINTF(ProfilerFile,
        "%lu %11.3f %11.2f %15d %15d %12.6f  %12.6f  %12.6f\n",
        N, MILLIONTH * dataset, BILLIONTH * rate,
        threadBlockSize, numThreadBlocks, avgtime, mintime, maxtime);
  }
}

void gpuProfilerCloseFile(void)
{
  if (ProfilerFile != NULL) {
    if (fclose(ProfilerFile) != 0) {
      perror("Error closing GPU profiler file");
    }
    ProfilerFile = NULL;
  }
}

#endif
