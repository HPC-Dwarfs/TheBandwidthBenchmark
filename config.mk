# Supported: GCC, CLANG, ICX, NVCC, HIP
TOOLCHAIN ?= HIP
# No effect when using NVCC or HIP toolchain
ENABLE_OPENMP ?= true
ENABLE_LIKWID ?= false

#Feature options
# 4GB dataset for desktop systems
OPTIONS  =  -DSIZE=500000000ull
# 40GB dataset for server systems
# OPTIONS  =  -DSIZE=1250000000ull
OPTIONS +=  -DNTIMES=1000
# Enable to enforce AVX512 streaming stores
#OPTIONS +=  -DAVX512_INTRINSICS
OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
