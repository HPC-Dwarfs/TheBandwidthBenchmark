CC   = hipcc
LD = $(CC)

VERSION   = --version
HIPFLAGS = --offload-arch=gfx900
HIPFLAGS += --offload-arch=gfx906
HIPFLAGS += --offload-arch=gfx908
HIPFLAGS += --offload-arch=gfx90a
HIPFLAGS += --offload-arch=gfx942
CPUFLAGS  = -O3 -pipe 
CFLAGS    = -O3 $(HIPFLAGS) -fgpu-rdc
LFLAGS    = -lhiprand
DEFINES   = -D_GNU_SOURCE
DEFINES   += -D__HIP_PLATFORM_AMD__
INCLUDES  =
LIBS      =
