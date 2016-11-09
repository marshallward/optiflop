#CC=gcc
#CFLAGS=-march=corei7-avx -O2 -fargument-noalias
CC=icc
CFLAGS=-xavx -O2

all: xyzp dxyzp

xyzp: xyzp.o dummy.o
	$(CC) -o $@ -lrt $^

dxyzp: dxyzp.o ddummy.o
	$(CC) -o $@ -lrt $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) xyzp dxypz *.o
