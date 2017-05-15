TARGET=gcc
ARCH=sandybridge

# TODO: Clean this up (maybe templates?)
ifeq ($(ARCH), sandybridge)
	ARCH=corei7-avx
else ifeq ($(ARCH), haswell)
	ARCH=core-avx2
else # Roll the dice...
	ARCH=native
endif

ifeq ($(TARGET), intel)
	CC=icc
	CFLAGS=-O2 -xavx
else # gnu
	CC=gcc
	CFLAGS=-march=$(ARCH) -O3
endif

LDFLAGS=-lrt

all: flop

flop: flop.o avx.o timer.o axpy.o dummy.o
	$(CC) $(LDFLAGS) -fopenmp -pthread -o $@ $^

flop.o: flop.c avx.h
	$(CC) -c -pthread -o $@ $<

avx.o: avx.c timer.h
	$(CC) $(CFLAGS) -c -o $@ $<

axpy.o: axpy.c timer.h
	$(CC) $(CFLAGS) -funroll-loops --param max-unroll-times=2 -c -o $@ $<

timer.o: timer.c timer.h
	$(CC) -O2 -c -o $@ $<

dummy.o: dummy.c
	$(CC) -c -o $@ $<

clean:
	$(RM) flop flop.o avx.o timer.o axpy.o
