# TODO: Clean this up (maybe templates?)
ifeq ($(ARCH), sandybridge)
	GCC_ARCH=corei7-avx
else ifeq ($(ARCH), haswell)
	GCC_ARCH=core-avx2
else ifeq ($(ARCH), knl)
	GCC_ARCH=knl
else # Roll the dice...
	GCC_ARCH=native
endif

LDFLAGS=-lrt -lm
ARCH=generic

all: gcc

# Platforms

gcc: CC=gcc -std=gnu99 -g -I.
gcc: CFLAGS=-march=$(GCC_ARCH) -O3 -fno-builtin
gcc: LOOP=-funroll-loops --param max-unroll-times=2
gcc: flop

intel: CC=icc -std=gnu99 -g -I.
intel: CFLAGS=-xavx -O3 -fno-builtin
intel: flop

stream: CC=icc -std=gnu99 -I.
stream: CFLAGS=-xavx -O3 -mcmodel=medium -ffreestanding -qopt-streaming-stores always
stream: flop

knl: CC=icc -I.
knl: CFLAGS=-std=gnu99 -xMIC-AVX512 -O3 -DBYTEALIGN=64
knl: flop

power: CC=gcc -std=gnu99 -g -I.
power: CFLAGS=-mpowerpc -O3 -fno-builtin
power: LOOP=-funroll-loops --param max-unroll-times=2
power: flop

# Build rules

flop: main.o bench.o stopwatch.o stopwatch_tsc.o avx.o axpy.o dummy.o
	$(CC) $(LDFLAGS) -fopenmp -pthread -o $@ $^

main.o: main.c avx.h axpy.h bench.h
	$(CC) -c -pthread -o $@ $<

bench.o: bench.c bench.h
	$(CC) -c -pthread -o $@ $<

avx.o: $(ARCH)/avx.c avx.h bench.h stopwatch.h
	$(CC) $(CFLAGS) -c -o $@ $<

axpy.o: axpy.c axpy.h bench.h stopwatch.h
	$(CC) $(CFLAGS) $(LOOP) -c -o $@ $<

dummy.o: dummy.c
	$(CC) -c -o $@ $<

stopwatch.o: stopwatch.c stopwatch.h
	$(CC) -O2 -c -o $@ $<

stopwatch_tsc.o: $(ARCH)/stopwatch_tsc.c
	$(CC) -O2 -c -o $@ $<

clean:
	$(RM) flop *.o *.h.gch
