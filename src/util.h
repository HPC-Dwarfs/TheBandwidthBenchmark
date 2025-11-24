/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef UTIL_H_
#define UTIL_H_

#define HLINE "------------------------------------------------------------------------\n"

#define BANNER                                                                           \
  "_|                            _|_|_|                                  _|  "           \
  "      \n"                                                                             \
  "_|_|_|    _|      _|      _|  _|    _|    _|_|    _|_|_|      _|_|_|  "               \
  "_|_|_|    \n"                                                                         \
  "_|    _|  _|      _|      _|  _|_|_|    _|_|_|_|  _|    _|  _|        _|  "           \
  "  _|  \n"                                                                             \
  "_|    _|    _|  _|  _|  _|    _|    _|  _|        _|    _|  _|        _|  "           \
  "  _|  \n"                                                                             \
  "_|_|_|        _|      _|      _|_|_|      _|_|_|  _|    _|    _|_|_|  _|  "           \
  "  _|  \n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#define DEBUG_MESSAGE debug_printf
#define FPRINTF(...)                                                                     \
  if (fprintf(__VA_ARGS__) < 0) {                                                        \
    printf("%s:%d Error writing to file\n", __FILE__, __LINE__);                         \
  }

#endif
