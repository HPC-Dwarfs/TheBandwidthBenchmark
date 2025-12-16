CC   = gcc
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -fopenmp
endif

ifeq ($(strip $(DATA_TYPE)),SP)
    DEFINES +=  -DPRECISION=1
else
    DEFINES +=  -DPRECISION=2
endif

VERSION  = --version
CFLAGS   = -O3 -ffast-math -march=native -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP)
DEFINES  += -D_GNU_SOURCE
INCLUDES =
LIBS     =
