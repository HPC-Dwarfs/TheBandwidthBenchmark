CC   = hipcc
LD   = $(CC)

ifeq ($(strip $(DATA_TYPE)),SP)
    DEFINES +=  -DPRECISION=1
else
    DEFINES +=  -DPRECISION=2
endif

# TARGETING SPECIFIC GPU ARCHITECTURES
# native: Good for local compilation
# gfx1030: AMD Radeon RX 6900 XT
# gfx942: MI300A / MI300X
# gfx90a: MI250X
# You can list multiple targets: --offload-arch=gfx90a --offload-arch=gfx942
HIP_ARCH  = --offload-arch=native

# TUNING FLAGS
# -munsafe-fp-atomics: dramatically speeds up atomicAdd on floats (similar to CUDA fast-math atomics)
HIPFLAGS  = $(HIP_ARCH)

VERSION   = --version
CPUFLAGS  = -O3 -pipe
CFLAGS    = -O3 $(HIPFLAGS) $(CPUFLAGS)
LFLAGS    =
DEFINES   += -D_GNU_SOURCE
DEFINES   += -D__HIP_PLATFORM_AMD__ -D_HIP
INCLUDES  = -I/opt/rocm-7.1.1/include/hiprand
LIBS      =
