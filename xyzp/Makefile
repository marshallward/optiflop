#CC=gcc
#CFLAGS=-march=corei7-avx -O2 -fargument-noalias
CC=icc
CFLAGS=-xavx -O2

all: xyzp

xyzp: xyzp.o dummy.o
	$(CC) -o $@ -lrt $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) xyzp xyzp.o dummy.o
