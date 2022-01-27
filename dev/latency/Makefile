CC = gcc
CPPFLAGS =
#CPPFLAGS = -DUSE_AVX512
CFLAGS = -g -O3 -mavx -mfma
LIBS = -lm

all: latency

latency: latency.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean
clean:
	$(RM) latency latency.o
