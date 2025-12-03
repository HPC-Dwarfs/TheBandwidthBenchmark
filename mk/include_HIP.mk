CC   = hipcc
LD   = $(CC)

VERSION   = --version
HIPFLAGS  = --offload-arch=native
CPUFLAGS  = -O3 -pipe
CFLAGS    = -O3 $(HIPFLAGS) $(CPUFLAGS)
LFLAGS    =
DEFINES   = -D_GNU_SOURCE
DEFINES   += -D__HIP_PLATFORM_AMD__ -D_HIP
INCLUDES  =
LIBS      =
