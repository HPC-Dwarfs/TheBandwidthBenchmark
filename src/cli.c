
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli.h"
#include "constants.h"
#include "util.h"

int BenchmarkType   = WS;
bool Sequential     = false;
int DataInitVariant = CONSTANT;
size_t N            = SIZE;
size_t Iterations   = NTIMES;

#ifdef _NVCC
int CUDA_DEVICE             = 0;
int THREAD_BLOCK_PER_SM     = 1024;
int THREAD_BLOCK_SIZE       = 2;
int THREAD_BLOCK_SIZE_SET   = 0;
int THREAD_BLOCK_PER_SM_SET = 0;
int VecVariant   = VEC0;
#endif

void parseArguments(int argc, char **argv)
{
  int co;
  opterr = 0;

  while ((co = getopt(argc, argv, "hm:s:n:i:d:t:b:v:")) != -1) {
    switch (co) {
    case 'h': {
      printf(HELPTEXT);
      exit(EXIT_SUCCESS);
      break;
    }

    case 'm': {
      if (strcmp(optarg, "ws") == 0) {
        BenchmarkType = WS;
      } else if (strcmp(optarg, "tp") == 0) {
        BenchmarkType = TP;
        Sequential    = false;
      } else if (strcmp(optarg, "seq") == 0) {
        BenchmarkType = SQ;
        Sequential    = true;
      } else {
        printf("Unknown bench type %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;
    }

    case 's': {
      char *end;
      errno = 0;
      N     = strtol(optarg, &end, INT_BASE);
      N     = (N * GIGA) / (4 * sizeof(double)); // Convert GB to number of doubles
      if (*end != '\0' || errno != 0) {
        FPRINTF(stderr, "Invalid numeric value for -s: %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;
    }

    case 'n': {
      char *end;
      errno      = 0;
      Iterations = strtol(optarg, &end, INT_BASE);
      if (*end != '\0' || errno != 0) {
        FPRINTF(stderr, "Invalid numeric value for -n: %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;
    }

    case 'i': {
      if (strcmp(optarg, "constant") == 0) {
        DataInitVariant = CONSTANT;
      } else if (strcmp(optarg, "random") == 0) {
        DataInitVariant = RANDOM;
      } else {
        printf("Invalid data initialization type %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;
    }

#ifdef _NVCC
    case 'd': {
      char *end;
      errno          = 0;
      const long val = strtol(optarg, &end, 10);
      if (*end != '\0' || errno != 0 || val < 0 || val > INT_MAX) {
        fprintf(stderr, "Invalid CUDA device ID: %s\n", optarg);
        exit(1);
      }
      CUDA_DEVICE = (int)val;

      break;
    }

    case 't': {
      char *end;
      errno          = 0;
      const long val = strtol(optarg, &end, 10);
      if (*end != '\0' || errno != 0 || val < 0 || val > INT_MAX) {
        fprintf(stderr, "Invalid Thread Block Size: %s\n", optarg);
        exit(1);
      }
      THREAD_BLOCK_SIZE     = (int)val;
      THREAD_BLOCK_SIZE_SET = 1;
      break;
    }

    case 'b': {
      char *end;
      errno          = 0;
      const long val = strtol(optarg, &end, 10);
      if (*end != '\0' || errno != 0 || val < 0 || val > INT_MAX) {
        fprintf(stderr, "Invalid Thread Blocks per SM: %s\n", optarg);
        exit(1);
      }
      THREAD_BLOCK_PER_SM     = (int)val;
      THREAD_BLOCK_PER_SM_SET = 1;
      break;
    }

    case 'v': {
      char *end;
      errno          = 0;
      const long val = strtol(optarg, &end, 10);

      if (*end != '\0' || errno != 0) {
        fprintf(stderr, "Invalid vectorization argument for -v: %s\n", optarg);
        exit(EXIT_FAILURE);
      }

      if (val == 0) {
        VecVariant = VEC0;
      } else if (val == 2) {
        VecVariant = VEC2;
      } else if (val == 4) {
        VecVariant = VEC4;
      } else {
        fprintf(stderr, "Unknown vectorization width: %ld. Allowed: 0, 2, 4\n", val);
        exit(EXIT_FAILURE);
      }

      break;
    }

#endif

    case '?': {
      if (optopt == 'c') {
        FPRINTF(stderr, "Option -%c requires an argument.\n", optopt);
      } else if (isprint(optopt)) {
        FPRINTF(stderr, "Unknown option `-%c'.\n", optopt);
      } else
        FPRINTF(stderr, "Unknown option character `\\x%x'.\n", optopt);
      exit(1);
    }

    default:
      abort();
    }
  }

  for (int index = optind; index < argc; index++) {
    printf("Non-option argument %s\n", argv[index]);
  }
}
