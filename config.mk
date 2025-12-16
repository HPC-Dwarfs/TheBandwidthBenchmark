# Supported: GCC, CLANG, ICX, NVCC, HIP
TOOLCHAIN ?= GCC
# Supported: true or false
ENABLE_OPENMP ?= true
# Supported: true or false
ENABLE_LIKWID ?= false
# Supported: SP or DP
DATA_TYPE ?= DP

#Feature options
# 4GB dataset for desktop systems
OPTIONS  =  -DSIZE=125000000ull
# 40GB dataset for server systems
# OPTIONS  =  -DSIZE=1250000000ull
OPTIONS +=  -DNTIMES=10
# Enable to enforce AVX512 streaming stores
#OPTIONS +=  -DAVX512_INTRINSICS
OPTIONS +=  -DARRAY_ALIGNMENT=64
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
