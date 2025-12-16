CC   = clang
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -fopenmp
# Uncomment for homebrew libomp on MacOS
# OPENMP   = -Xpreprocessor -fopenmp
# LIBS     = -L/opt/homebrew/opt/libomp/lib -lomp
endif

ifeq ($(strip $(DATA_TYPE)),SP)
    DEFINES +=  -DPRECISION=1
else
    DEFINES +=  -DPRECISION=2
endif

VERSION  = --version
CFLAGS   = -O3 -ffast-math -std=c99 $(OPENMP)
#CFLAGS   = -Ofast -fnt-store=aggressive  -std=c99 $(OPENMP) #AMD CLANG
LFLAGS   = $(OPENMP)
DEFINES  += -D_GNU_SOURCE
INCLUDES =
# Uncomment for homebrew libomp on MacOS
# INCLUDES = -I/opt/homebrew/opt/libomp/include
