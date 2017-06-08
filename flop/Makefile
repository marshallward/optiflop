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

LDFLAGS=-lrt

all: gcc

gcc: CC=gcc -std=gnu99
gcc: CFLAGS=-march=$(GCC_ARCH) -O3
gcc: LOOP=-funroll-loops --param max-unroll-times=2
gcc: flop

intel: CC=icc -std=gnu99
intel: CFLAGS=-xavx -O3
intel: flop

knl: CC=icc
knl: CFLAGS=-xMIC-AVX512 -O3
knl: flop

flop: main.o bench.o stopwatch.o avx.o axpy.o dummy.o
	$(CC) $(LDFLAGS) -fopenmp -pthread -o $@ $^

main.o: main.c bench.h
	$(CC) -c -pthread -o $@ $<

bench.o: bench.c bench.h
	$(CC) -c -pthread -o $@ $<

avx.o: avx.c stopwatch.h
	$(CC) $(CFLAGS) -c -o $@ $<

axpy.o: axpy.c stopwatch.h
	$(CC) $(CFLAGS) $(LOOP) -c -o $@ $<

dummy.o: dummy.c
	$(CC) -c -o $@ $<

stopwatch.o: stopwatch.c stopwatch.h
	$(CC) -O2 -c -o $@ $<

clean:
	$(RM) flop *.o *.h.gch
