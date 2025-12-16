CC   = nvcc
LD = $(CC)

ifeq ($(strip $(DATA_TYPE)),SP)
    DEFINES +=  -DPRECISION=1
else
    DEFINES +=  -DPRECISION=2
endif

VERSION   = --version
# For A100 GPUs, Ampere GA100 arch target
NVCCFLAGS = -gencode arch=compute_80,code=sm_80
#For A40 GPUs, Ampere GA102 arch target
NVCCFLAGS += -gencode arch=compute_86,code=sm_86
# For H100 GPUs, although compute_90a and sm_90a is recommended for Hopper Arch
# https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html#building-applications-with-hopper-support
NVCCFLAGS += -gencode arch=compute_90,code=sm_90 
NVCCFLAGS += -Xcompiler -rdynamic --generate-line-info -Wno-deprecated-gpu-targets
CPUFLAGS  = -O3 -pipe 
CFLAGS    = -O3 $(NVCCFLAGS) --compiler-options="$(CPUFLAGS)"
LFLAGS    = -lcuda
DEFINES   += -D_GNU_SOURCE
DEFINES   += -D_NVCC
INCLUDES  =
LIBS      =
