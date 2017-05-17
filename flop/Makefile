# TODO: Clean this up (maybe templates?)
ifeq ($(ARCH), sandybridge)
	GCC_ARCH=corei7-avx
else ifeq ($(ARCH), haswell)
	GCC_ARCH=core-avx2
else # Roll the dice...
	GCC_ARCH=native
endif

LDFLAGS=-lrt

all: gcc

gcc: CC=gcc
gcc: CFLAGS=-march=$(GCC_ARCH) -O3
gcc: LOOP=-funroll-loops --param max-unroll-times=2
gcc: flop

intel: CC=icc
intel: CFLAGS=-xavx -O2
intel: flop

flop: flop.o avx.o stopwatch.o axpy.o dummy.o
	$(CC) $(LDFLAGS) -fopenmp -pthread -o $@ $^

flop.o: flop.c avx.h
	$(CC) -c -pthread -o $@ $<

avx.o: avx.c stopwatch.h
	$(CC) $(CFLAGS) -c -o $@ $<

axpy.o: axpy.c stopwatch.h
	$(CC) $(CFLAGS) $(LOOP) -c -o $@ $<

stopwatch.o: stopwatch.c stopwatch.h
	$(CC) -O2 -c -o $@ $<

dummy.o: dummy.c
	$(CC) -c -o $@ $<

clean:
	$(RM) flop flop.o avx.o stopwatch.o axpy.o dummy.o
