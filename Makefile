CC=gcc
CFLAGS=-march=corei7-avx -O2 -lrt -funroll-loops

all: avx_add avx_mac

avx_add: avx_add.c
	$(CC) $(CFLAGS) -o $@ $^

avx_mac: avx_mac.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	$(RM) avx_add avx_mac
