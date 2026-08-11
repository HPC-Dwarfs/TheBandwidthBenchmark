/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of TheBandwidthBenchmark.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef METRICS_H_
#define METRICS_H_

/* GPU telemetry sampler (NVML on CUDA, ROCm SMI on HIP).
 *
 * This header is C++-only and intended for inclusion from kernels.cu (outside
 * any extern "C" block). The C-visible per-kernel record (KernelMetricsType)
 * lives in profiler.h so that profiler.c does not pull in <thread>/<atomic>. */

#ifdef NVML

#ifndef __cplusplus
#error "metrics.h must be included from C++ translation units only"
#endif

#include "portability.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#ifndef _HIP
#include <nvml.h>
#else
#include <rocm_smi/rocm_smi.h>
#endif

/* Averaged metrics over the most recent start()/stop() window. */
struct MetricsAvg {
  double power;   /* W   */
  double clock;   /* MHz */
  double elapsed; /* seconds between start() and stop() */
};

/* Background-thread sampler. Usage:
 *
 *   GpuMonitor mon;
 *   if (!mon.init(device_id)) return 1;
 *   mon.start();              // resets accumulators, spawns thread
 *   ... run kernel iterations ...
 *   mon.stop();               // joins thread
 *   MetricsAvg a = mon.averages();
 *   ...
 *   mon.shutdown();           // tear down NVML / ROCm SMI
 *
 * start()/stop() may be called repeatedly between init() and shutdown(). */
struct GpuMonitor {
  double             totalPower;
  double             totalClock;
  int                samples;
  double             elapsed;

  std::atomic<bool>  _active;
  std::thread        _worker;
  std::chrono::steady_clock::time_point _start_tp;

#ifndef _HIP
  nvmlDevice_t _nvml_dev;
#else
  uint32_t     _rsmi_idx;
#endif

  GpuMonitor() : totalPower(0.0), totalClock(0.0), samples(0), elapsed(0.0)
  {
    _active.store(false);
  }

  inline bool init(int device_id)
  {
#ifndef _HIP
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to init NVML: " << nvmlErrorString(r) << std::endl;
      return false;
    }
    r = nvmlDeviceGetHandleByIndex(device_id, &_nvml_dev);
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to get NVML device: " << nvmlErrorString(r) << std::endl;
      return false;
    }
#else
    rsmi_status_t r = rsmi_init(0);
    if (r != RSMI_STATUS_SUCCESS) {
      const char *err_str;
      rsmi_status_string(r, &err_str);
      std::cerr << "Failed to init ROCm SMI: " << err_str << std::endl;
      return false;
    }
    _rsmi_idx = (uint32_t)device_id;
#endif
    return true;
  }

  inline void start()
  {
    totalPower = 0.0;
    totalClock = 0.0;
    samples    = 0;
    elapsed    = 0.0;
    _start_tp  = std::chrono::steady_clock::now();
    _active.store(true);

    _worker = std::thread([this]() {
      while (_active.load()) {
#ifndef _HIP
        unsigned int power_mW = 0, clock_MHz = 0;
        if (nvmlDeviceGetPowerUsage(_nvml_dev, &power_mW) == NVML_SUCCESS)
          totalPower += power_mW / 1000.0;
        if (nvmlDeviceGetClockInfo(_nvml_dev, NVML_CLOCK_SM, &clock_MHz) == NVML_SUCCESS)
          totalClock += clock_MHz;
#else
        uint64_t           power_uW = 0;
        RSMI_POWER_TYPE    power_type;
        rsmi_frequencies_t freqs;
        if (rsmi_dev_power_get(_rsmi_idx, &power_uW, &power_type) == RSMI_STATUS_SUCCESS)
          totalPower += (double)power_uW / 1.0e6;
        if (rsmi_dev_gpu_clk_freq_get(_rsmi_idx, RSMI_CLK_TYPE_SYS, &freqs) ==
            RSMI_STATUS_SUCCESS)
          totalClock += (double)freqs.frequency[freqs.current] / 1.0e6;
#endif
        samples++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  inline void stop()
  {
    _active.store(false);
    if (_worker.joinable()) _worker.join();
    auto end_tp = std::chrono::steady_clock::now();
    elapsed     = std::chrono::duration<double>(end_tp - _start_tp).count();
  }

  inline MetricsAvg averages() const
  {
    MetricsAvg a = { 0.0, 0.0, elapsed };
    if (samples > 0) {
      a.power = totalPower / samples;
      a.clock = totalClock / samples;
    }
    return a;
  }

  inline void shutdown()
  {
#ifndef _HIP
    nvmlShutdown();
#else
    rsmi_shut_down();
#endif
  }
};

#endif /* NVML */

#endif /* METRICS_H_ */
