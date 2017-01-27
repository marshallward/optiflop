/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <time.h>       /* timespec, clock_gettime */

#include "timer.h"
#include "avx.h"


int main(int argc, char *argv[])
{
    float result;
    double runtime;

    runtime = avx_add();

    printf("avx_add\n");
    printf("-------\n");
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (8 registers / iteration) */
    printf("gflops: %.12f\n", N * 8 * 8 / (runtime * 1e9));

    printf("\n");

    runtime = avx_mac();

    printf("avx_mac\n");
    printf("-------\n");
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (8 registers / iteration) */
    printf("gflops: %.12f\n", N * 8 * 48 / (runtime * 1e9));

    return 0;
}
