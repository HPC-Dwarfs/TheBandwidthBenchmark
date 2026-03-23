# AI Agent Instructions for TheBandwidthBenchmark

This file contains properties, build instructions, and common pitfalls for TheBandwidthBenchmark project to help AI agents (like Claude, Cursor, Roo, etc.) assist effectively.

## Project Overview
- **Purpose**: Measure maximum sustained main memory bandwidth of CPU and GPU systems, as well as the complete memory hierarchy using sequential or parallel throughput execution.
- **Languages**: C, C++, CUDA, Make.
- **Key Kernels**: init, sum, copy, update, triad, daxpy, striad, sdaxpy.

## Build System & Toolchains
- The project uses `make` as the build system.
- **Supported Compilers**: GCC, Clang, Intel ICC/ICX, NVCC (for GPU).
- **Configuration**: Managed in `config.mk`. Edit this file to change the default toolchain (`TOOLCHAIN`), enable/disable OpenMP (`ENABLE_OPENMP`), enable/disable LIKWID (`ENABLE_LIKWID`), and define problem sizes (e.g., `-DSIZE=...`).
- **Commands**:
  - `make` - Build the current toolchain (set in `config.mk`).
  - `make clean` - Clean intermediate build results for the active toolchain.
  - `make distclean` - Clean build results for all toolchains.
  - `make asm` - Generate assembler files.
  - `make format` - Format code using `clang-format`.
  - `make plot` / `make plot_dataset` - Generate gnuplot visualizations from outputs in `./dat`.

## Project Structure & Artifacts
- Source code is located in `src/` and `core/`.
- Intermediate build objects and binaries are placed in `./build/<TOOLCHAIN>/`.
- Benchmark parameter sweeps outputs (modes `-m tp` and `-m seq`) are generated in the `./dat` directory.
- Plots are output to the `./plot` directory.

## Running the Benchmark
- Binary naming pattern: `./bwBench-<TOOLCHAIN>`, e.g., `./bwbench-GCC`.
- **Modes** (`-m <mode>`):
  - `ws`: Worksharing (default)
  - `seq`: Sequential (sweeps over array sizes, can take a long time)
  - `tp`: Throughput (sweeps over sizes with OpenMP, can take a long time)
- **Thread Pinning**: CPU threading is best controlled with `likwid-pin` (e.g., `likwid-pin -C 0-3 ./bwbench-GCC`). If LIKWID is not available, use the command line argument `-p compact`.

## GPU Specifics (CUDA)
- Set `TOOLCHAIN=NVCC` in `config.mk` for GPU builds.
- Configure `-d <ID>` to run on a specific GPU.
- Tuning macros: `THREADBLOCKSIZE` (default 1024) and `THREADBLOCKPERSM` (default 2). `THREADBLOCKSIZE` always takes precedence for kernel launch configuring.

## Caveats & Pitfalls (Critical to Remember)
- **Intel ICX/ICPX Compilers**:
  - Non-Temporal (NT) Stores: If using `-qopt-streaming-stores=always`, DO NOT use the `-ffreestanding` flag. It will break the generation of NT instructions and fallback to `__libirc_nontemporal_store@PLT` library calls.
  - In Throughput (`tp`) mode with OpenMP, the icx/icpx compiler does NOT respect the `nontemporal()` clause with the OpenMP `simd` directive.
- **Measuring Cache Hierarchy**: Do NOT use non-temporal (NT) streaming stores if the goal is to measure cache hierarchy bandwidth using Sequential (`seq`) or Throughput (`tp`) mode. This would bypass the cache and invalidate the measurement.
- **OpenMP Overheads**: If performance is lower than expected, check the problem size. The default size is 4GB to cover barrier costs on many-core systems. If overhead is too high, increase size using `-s <SIZE>` on CLI or `SIZE` in `config.mk`.
- **Makefile Context**: The Makefile only builds and acts on the *currently* set `TOOLCHAIN` in `config.mk`. To work with a different toolchain, ensure `config.mk` is updated first.
