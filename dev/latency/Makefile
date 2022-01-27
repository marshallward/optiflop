CC = gcc
#CPPFLAGS =
#CFLAGS = -g -O3 -mavx -mfma
CPPFLAGS = -DUSE_AVX512
CFLAGS = -g -O3 -mavx -mfma -mavx512f
LIBS = -lm

all: latency

latency: latency.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean
clean:
	$(RM) latency latency.o
