# SPDX-License-Identifier: Apache-2.0

.PHONY: clean

TARGET = bench
CC  ?= gcc
LD  := $(CC)

SOURCES = \
    hal/hal.c \
    bench.c \
    toom3.c \
    gmp_mul_benchmark.c \
	ntt_zetas.c \
	ntt.S

CFLAGS := \
	-Wall \
	-Wextra \
	-Werror=unused-result \
	-Wpedantic \
	-Werror \
	-Wmissing-prototypes \
	-Wshadow \
	-Wpointer-arith \
	-Wredundant-decls \
	-Wno-long-long \
	-Wno-unknown-pragmas \
	-Wno-unused-command-line-argument \
	-O3 \
	-fomit-frame-pointer \
	-std=c99 \
	-pedantic \
	-Ihal \
	-MMD \
	$(CFLAGS)

ifeq ($(CYCLES),PMU)
	CFLAGS += -DPMU_CYCLES
endif

ifeq ($(CYCLES),PERF)
	CFLAGS += -DPERF_CYCLES
endif

ifeq ($(CYCLES),MAC)
	CFLAGS += -DMAC_CYCLES
endif

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) modmul_asm.o -o $(TARGET) -lgmp

clean:
	-$(RM) -rf $(TARGET) *.d