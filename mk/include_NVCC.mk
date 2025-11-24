CC   = nvcc
LD = $(CC)

VERSION   = --version
NVCCFLAGS = -gencode arch=compute_80,code=sm_80
NVCCFLAGS += -gencode arch=compute_86,code=sm_86
NVCCFLAGS += -gencode arch=compute_90,code=sm_90 
NVCCFLAGS += -Xcompiler -rdynamic --generate-line-info -Wno-deprecated-gpu-targets
CPUFLAGS  = -O3 -pipe 
CFLAGS    = -O3 $(NVCCFLAGS) --compiler-options="$(CPUFLAGS)"
LFLAGS    = -lcuda
DEFINES   = -D_GNU_SOURCE
DEFINES   += -D_NVCC
INCLUDES  =
LIBS      =
