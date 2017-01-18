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

all: avx avx_mac

avx: avx.o gettime.o
	$(CC) $(LDFLAGS) -o $@ $^

avx.o: avx.c gettime.h
	$(CC) $(CFLAGS) -c -o $@ $<

gettime.o: gettime.c gettime.h
	$(CC) -O2 -c -o $@ $<

avx_add: avx_add.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

avx_mac: avx_mac.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	$(RM) avx_add avx_mac avx avx.o gettime.o
