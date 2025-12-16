CC   = icx
LD = $(CC)

ifeq ($(strip $(DATA_TYPE)),SP)
    DEFINES +=  -DPRECISION=1
else
    DEFINES +=  -DPRECISION=2
endif

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -qopenmp
endif

ifeq ($(ENABLE_LTO),true)
FAST_WORKAROUND = -ipo -O3 -static -fp-model=fast
else
FAST_WORKAROUND = -O3 -static -fp-model=fast
endif

VERSION  = --version
CFLAGS   = $(FAST_WORKAROUND) -xHost -qopt-zmm-usage=high -std=c99 -ffreestanding $(OPENMP)
CFLAGS   += -Wimplicit-const-int-float-conversion -Wno-unused-command-line-argument
LFLAGS   = $(OPENMP)
DEFINES  += -D_GNU_SOURCE
INCLUDES =
LIBS     =
