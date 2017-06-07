CC=gcc
CFLAGS=-march=native -O3 -funroll-loops --param max-unroll-times=2
#CC=icc
#CFLAGS=-xavx -O3

all: axpy

axpy: axpy.o dummy.o
	$(CC) -o $@ -lrt $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) axpy axpy.o dummy.o
