# Makefile — SH-WFS Centroiding C Library
# =========================================
# Targets:
#   make           → build shared library (auto-detects OpenMP)
#   make debug     → debug build with sanitizers
#   make bench     → standalone benchmark binary
#   make clean

CC      := gcc
CFLAGS  := -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS := -shared -lm

# Auto-detect OpenMP support
OMP_TEST := $(shell echo 'int main(){}' | $(CC) -fopenmp -x c - -o /dev/null 2>&1)
ifeq ($(OMP_TEST),)
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
    $(info OpenMP: ENABLED)
else
    $(info OpenMP: NOT available — building without parallelism)
endif

TARGET  := libcentroid.so
SRCS    := centroid.c
HDRS    := centroid.h

.PHONY: all debug bench clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)
	@echo "Built $(TARGET)"

# Debug build with AddressSanitizer
debug: CFLAGS := -O0 -g -fsanitize=address,undefined -Wall -Wextra -fPIC
debug: LDFLAGS := -shared -lm -fsanitize=address,undefined
debug: $(TARGET)

# Standalone benchmark / smoke-test binary
bench: centroid_bench.c $(SRCS) $(HDRS)
	$(CC) -O3 -march=native -fopenmp -Wall \
	    centroid_bench.c centroid.c -lm -o centroid_bench
	@echo "Built centroid_bench — run with ./centroid_bench"

clean:
	rm -f $(TARGET) centroid_bench *.o
