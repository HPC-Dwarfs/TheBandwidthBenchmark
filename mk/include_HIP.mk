CC   = hipcc
LD = $(CC)

VERSION   = --version
HIPFLAGS  = -O3
CPUFLAGS  = -O3 -pipe 
CFLAGS    = $(HIPFLAGS) $(CPUFLAGS)
LFLAGS    = 
DEFINES   = -D_GNU_SOURCE
DEFINES   += -D_HIP
INCLUDES  =
LIBS      =