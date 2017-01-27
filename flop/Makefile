TARGET=gcc
ARCH=haswell

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
	CFLAGS=-O2
else # gnu
	CC=gcc
	CFLAGS=-march=$(ARCH) -O2 -funroll-loops
endif

LDFLAGS=-lrt

all: flop

flop: flop.o avx.o timer.o
	$(CC) $(LDFLAGS) -o $@ $^

flop.o: flop.c avx.h
	$(CC) $(CFLAGS) -c -o $@ $<

avx.o: avx.c timer.h
	$(CC) $(CFLAGS) -c -o $@ $<

timer.o: timer.c timer.h
	$(CC) -O2 -c -o $@ $<

clean:
	$(RM) avx avx.o timer.o
