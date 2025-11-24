# The Bandwidth Benchmark

This is a collection of simple streaming kernels. Its primary purpose is to
measure the maximum sustained main memory bandwidth of CPU and GPU systems.
It also offers a mode to measure the complete memory
hierarchy using sequential or parallel throughput execution.
You may want to take a look at
<https://github.com/RRZE-HPC/TheBandwidthBenchmark/wiki> for a collection of
results created using TheBandwidthBenchmark.

## Overview

The benchmark is heavily inspired by John McCalpin's
<https://www.cs.virginia.edu/stream/> benchmark.

It contains the following streaming kernels with corresponding data access
pattern (Notation: S - store, L - load, WA - write allocate). All variables are
vectors, s is a scalar:

- init (S1, WA): Initilize an array: `a = s`. Store only.
- sum (L1): Vector reduction: `s += a`. Load only.
- copy (L1, S1, WA): Classic memcopy: `a = b`.
- update (L1, S1): Update vector: `a = a * scalar`. Also load + store but
  without write allocate.
- triad (L2, S1, WA): Stream triad: `a = b + c * scalar`.
- daxpy (L2, S1): Daxpy: `a = a + b * scalar`.
- striad (L3, S1, WA): Schoenauer triad: `a = b + c * d`.
- sdaxpy (L3, S1): Schoenauer triad without write allocate: `a = a + b * c`.

## Getting Started

To build and run **The Bandwidth Benchmark**, you only need a compiler and GNU
make.

1. **Install a supported compiler**
   - GCC
   - Clang
   - Intel ICC/ICX
   - NVCC (for CUDA builds)

2. **Clone the repository**

   ```sh
   git clone https://github.com/RRZE-HPC/TheBandwidthBenchmark.git

   cd TheBandwidthBenchmark
   ```

3. **(Optional) Adjust configuration**

   Edit `config.mk` to change the default problem size, enable OpenMP, set GPU
   launch parameters, etc.

4. **Build**

   CPU: `make`

   GPU (set `TOOLCHAIN=NVCC` in `config.mk`): `make`

   See the full [Build](#build) section for more details.

5. **Usage**

   ```sh
   ./bwBench-<TOOLCHAIN>
   ```

   See the full [Usage](#usage) section for more details.

   Get _Help_ on command line arguments:

   ```sh
   ./bwBench-<TOOLCHAIN> -h
   ```

## Build

### CPU Build

1. Configure the tool chain and additional options in `config.mk`:

```make
# Supported: GCC, CLANG, ICC, ICX
TOOLCHAIN ?= GCC
ENABLE_OPENMP ?= true
ENABLE_LIKWID ?= false

#Feature options
# 4GB dataset for desktop systems
OPTIONS  =  -DSIZE=125000000ull
OPTIONS +=  -DNTIMES=100
# 40GB dataset for server systems
# OPTIONS  =  -DSIZE=1250000000ull
# OPTIONS +=  -DNTIMES=10
# Enable to enforce AVX512 streaming stores
#OPTIONS +=  -DAVX512_INTRINSICS
OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
```

The verbosity options enable detailed output about affinity settings, allocation
sizes, and timer resolution. If you uncomment `-DVERBOSE_AFFINITY` the processor
id every thread is currently scheduled on and the complete affinity mask for
every thread is printed.

_Notice:_ OpenMP involves significant overhead through barrier cost, especially
on systems with many memory domains. The default problem size is set to 4GB to
have enough work vs overhead. If you suspect that the result should be better
you may try to further increase the problem size. This can be done either by
change the `SIZE` define or using the command line option `-s <SIZE>`. To
compare to original stream results on X86 systems you have to ensure that
streaming store instructions are used. For the ICC tool chain this is the
default (Option `-qopt-streaming-stores=always`). If the configured tool chain
does not generate streaming stores you can comment out `-DAVX512_INTRINSICS` on
processors that support AVX512. This enables to use intrinsics instead of
compiler generated code.

- Build with:

```sh
make
```

You can build multiple tool chains in the same directory, but notice that the
Makefile is only acting on the one currently set. Intermediate build results are
located in the `./build/<TOOLCHAIN>` directory.

- Clean up intermediate build results for active tool chain, data files and plots with:

```sh
make clean
```

Clean all build results for all tool chains:

```sh
make distclean
```

- Optional targets:

Generate assembler:

```sh
make asm
```

The assembler files will also be located in the `./build/<TOOLCHAIN>` directory.

Reformat all source files using `clang-format` (only works if `clang-format` is
in your path):

```sh
make format
```

---

### GPU Build (NVIDIA CUDA)

When building bwBench with CUDA support, several compile-time parameters can be
tuned to optimize performance on different NVIDIA GPU architectures.

```make
# Supported: GCC, CLANG, ICX, NVCC
TOOLCHAIN ?= NVCC
ENABLE_OPENMP ?= false
ENABLE_LIKWID ?= false

OPTIONS  =  -DSIZE=120000000ull
OPTIONS +=  -DNTIMES=10
OPTIONS +=  -DARRAY_ALIGNMENT=64
OPTIONS +=  -DTHREADBLOCKSIZE=1024
OPTIONS +=  -DTHREADBLOCKPERSM=2
```

#### GPU Tuning Parameters

##### `THREADBLOCKSIZE` (default = 1024)

Defines the CUDA thread block size used for the benchmark kernels (i.e., the
number of threads per block).

- Typical values: 128, 256, 512, 1024
- Larger block sizes may improve occupancy for some architectures, but not all GPUs benefit equally.
- This parameter **has the highest priority** when determining the kernel launch configuration.

##### `THREADBLOCKPERSM` (default = 2)

Defines the _requested_ number of thread blocks per Streaming Multiprocessor
(SM). This allows you to control occupancy and tailor the workload to your
specific GPU.

- Example: setting `THREADBLOCKPERSM=2` requests that the runtime attempt to schedule two blocks per SM.

---

#### Interaction Between `THREADBLOCKSIZE` and `THREADBLOCKPERSM`

`THREADBLOCKSIZE` **always takes precedence** when determining kernel execution parameters:

1. The program first attempts to launch kernels with the given `THREADBLOCKSIZE`.
2. It then checks whether the GPU can support the requested `THREADBLOCKPERSM` with that block size.
3. **If the target `THREADBLOCKPERSM` is achievable** given hardware limits and the selected `THREADBLOCKSIZE`, the program uses it.
4. **If not**, the program automatically falls back to the _maximum feasible_ blocks per SM for the given thread block size.

## Usage

To run the benchmark call:

```sh
./bwBench-<TOOLCHAIN> [Command-Line-Arguments]
```

### Command Line Arguments

`NOTICE:` Command Line Arguments override the **OPTIONS** set in `config.mk`

| Option | Argument     | Description                                                                                                                 |
| ------ | ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `-h`   | —            | Show help text.                                                                                                             |
| `-m`   | `<type>`     | _(CPU only)_ Benchmark type. Valid values:<br>• `ws` — Worksharing (default)<br>• `tp` — Throughput<br>• `seq` — Sequential |
| `-s`   | `<long int>` | Size (in GB) of the allocated vectors.                                                                                      |
| `-n`   | `<long int>` | Number of iterations.                                                                                                       |
| `-i`   | `<type>`     | Data initialization type. Valid values:<br>• `constant` (default) <br>• `random`                                            |
| `-d`   | `<int>`      | _(GPU-enabled builds only)_ GPU ID on which the program should run. (default = 0)                                           |
| `-t`  | `<int>`      | _(GPU-enabled builds only)_ Thread Block Size (default = 1024)                                                              |
| `-b` | `<int>`      | _(GPU-enabled builds only)_ Thread Block per SM. (default = 2)                                                              |

In default mode the benchmark will output the results similar to the STREAM
benchmark. Results are validated.

---

### Sequential vs Throughput mode: Sweeping over a range of problem size

Apart from the default parallel work sharing mode with fixed problem size
TheBandwidthBenchmark also supports two modes with varying problem sizes:
sequential (call with `seq` mode option) and throughput (call with `tp` mode
option). These are intended for scanning the complete memory hierarchy instead
of only the main memory domain. See below for details on how to use those modes.

**NOTICE:** The `seq` and `tp` modes may take up to 30m or more, depending on
the system.

These 2 modes performs a sweep over different array sizes ranging from N = 100
until the **array size N** specified in `config.mk`.

- **Sequential** - Runs TheBandwidthBenchmark in sequential mode for all kernels. Command to run in sequential mode:

```sh
./bwBench-<TOOLCHAIN> -m seq
```

- **Throughput (Multi-threaded)** - Runs TheBandwidthBenchmark in multi-threaded mode for all kernels. Requires flag **ENABLE_OPENMP=true** in `config.mk`.
  Command to run in throughput mode:

```sh
./bwBench-<TOOLCHAIN> -m tp
```

Each of these modes output the results for each individual kernel.

The output files will be created in the `./dat` directory.

#### Visualizing the data from the Sequential/Throughput modes

`Required:` **Gnuplot 5.2+**

The user can visualize the outputs from the `./dat` directory using the provided
gnuplot scripts. The scripts are located in `./gnuplot_script` directory where a
bash file takes care of generating and executing the gnuplot commands. The plots
can then be found in the `./plot` directory.

There are 2 ways you can visualize the output:

- **Plotting Array Size (N) vs Bandwidth (MB/s)** - this mode creates plot with
  the Array Size (N) on x-axis and Bandwidth (MB/s) on y-axis. The Array size (N)
  will be the same for each kernel. Use this makefile command to generate this
  type of plot:

```sh
make plot
```

- **Plotting Dataset Size (MB) vs Bandwidth (MB/s)** - this mode creates plot
  with the Dataset Size (MB) on x-axis and Bandwidth (MB/s) on y-axis. The Dataset
  size (MB) will be the different for each kernel. For example the total dataset
  for Init kernel will be 4x times less than the total dataset size for the STriad
  kernel.

```sh
make plot_dataset
```

The script also generates a combined plot with bandwidths from all the kernels
into one plot.

---

### Thread pinning

For threaded execution it is recommended to control thread affinity. We
recommend to use `likwid-pin` for setting the number of threads used and to
control thread affinity:

```sh
likwid-pin -C 0-3 ./bwbench-GCC
```

Example output for threaded execution:

```txt
-------------------------------------------------------------
[pthread wrapper]
[pthread wrapper] MAIN -> 0
[pthread wrapper] PIN_MASK: 0->1  1->2  2->3
[pthread wrapper] SKIP MASK: 0x0
        threadid 140271463495424 -> core 1 - OK
        threadid 140271455102720 -> core 2 - OK
        threadid 140271446710016 -> core 3 - OK
OpenMP enabled, running with 4 threads
----------------------------------------------------------------------------
Function      Rate(MB/s)  Rate(MFlop/s)  Avg time     Min time     Max time
Init:          22111.53    -             0.0148       0.0145       0.0165
Sum:           46808.59    46808.59      0.0077       0.0068       0.0140
Copy:          30983.06    -             0.0207       0.0207       0.0208
Update:        43778.69    21889.34      0.0147       0.0146       0.0148
Triad:         34476.64    22984.43      0.0282       0.0278       0.0305
Daxpy:         45908.82    30605.88      0.0214       0.0209       0.0242
STriad:        37502.37    18751.18      0.0349       0.0341       0.0388
SDaxpy:        46822.63    23411.32      0.0281       0.0273       0.0325
----------------------------------------------------------------------------
Solution Validates
```

If `likwid-pin` is not available you can use the command line argument `-p
compact` to enable internal pinning using the `OMP_PLACES` pragma.


## Scaling runs

Apart from the highest sustained memory bandwidth also the scaling behavior
within memory domains is an important system property.

There is a helper script downloadable at
<https://github.com/RRZE-HPC/TheBandwidthBenchmark/wiki/util/extractResults.pl>
that creates a text result file from multiple runs that can be used as input to
plotting applications as gnuplot and xmgrace. This involves two steps: Executing
the benchmark runs and creating the data file.

To run the benchmark for different thread counts within a memory domain execute
(this assumes bash or zsh):

```sh
for nt in 1 2 4 6 8 10; do likwid-pin -q -C E:M0:$nt:1:2 ./bwbench-ICC > dat/emmy-$nt.txt; done
```

It is recommended to just use one thread per core in case the processor supports
hyperthreading. Use whatever stepping you like, here a stepping of two was used.
The `-q` option suppresses output from `likwid-pin`. Above line uses the
expression based syntax, on systems with hyperthreading enabled (check with,
e.g., `likwid-topology`) you have to skip the other hardware threads on each
core. For above system with 2 hardware threads per core this results in `-C
E:M0:$nt:1:2`, on a system with 4 hardware threads per core you would need `-C
E:M0:$nt:1:4`. The string before the dash (here emmy) can be arbitrary, but the
the extraction script expects the thread count after the dash. Also the file
ending has to be `.txt`. Please check with a text editor on some result files if
everything worked as expected.

To extract the results and output in a plot table format execute:

```sh
./extractResults.pl ./dat
```

The script will pick up all result files in the directory specified and create a
column format output file. In this case:

```txt
#nt     Init    Sum     Copy    Update  Triad   Daxpy   STriad  SDaxpy
1       4109    11900   5637    8025    7407    9874    8981    11288
2       8057    22696   11011   15174   14821   18786   17599   21475
4       15602   39327   21020   28197   27287   33633   31939   37146
6       22592   45877   29618   37155   36664   40259   39911   41546
8       28641   46878   35763   40111   40106   41293   41022   41950
10      33151   46741   38187   40269   39960   40922   40567   41606
```

Please be aware the single core memory bandwidth as well as the scaling behavior
depends on the frequency settings.

## Support for clangd language server

The Makefile will generate a `.clangd` configuration to correctly set all
options for the clang language server. This is only important if you use an
editor with LSP support and want to edit or explore the source code.
It is required to use GNU Make 4.0 or newer. While older make versions will
work, the generation of the `.clangd` configuration for the clang language
server will not work. The default Make version included in MacOS is 3.81! Newer make
versions can be easily installed on MacOS using the
[Homebrew](https://brew.sh/) package manager.

An alternative is to use [Bear](https://github.com/rizsotto/Bear), a tool that
generates a compilation database for clang tooling. This method also will enable
to jump to any definition without previously opened buffer. You have to build
TheBandwidthBenchmark one time with Bear as a wrapper:

```sh
bear -- make
```

## Caveats

A few known issues, based on the experience with specific compilers.

- Intel oneAPI DPC++/C++ Compiler 2023.2.0 (icx/icpx compiler):
  - NonTemporal Stores (aka Streaming Stores): We leave the choice to the user whether to use NT stores or not.
    - If the user wants to use NT stores using `-qopt-streaming-stores=always` compiler flag, then the user has to avoid using the `-ffreestanding` compiler flag. This will not generate NT instructions, but generates calls to `__libirc_nontemporal_store@PLT` in the assembly.
    - For the Througput mode with OpenMP, the icx/icpx compiler does not respect the `nontemporal()` clause with the OpenMP `simd` directive.

    It's recommended not to use NT stores if the user wants to measure cache hierarchy bandwidth using the Sequential or Throughput mode.
