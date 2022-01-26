#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#define MINTIME 0.1
//#define USE_AVX512

int main(int argc, char *argv[])
{
    struct timespec ts_start, ts_end;
    double time;
    __m256 reg;
    volatile __m256 out;

    long r_max;
    volatile long v;
    double freq;

    const __m256 add0 = _mm256_set1_ps((float) 1e-6);
    const __m256 mul0 = _mm256_set1_ps((float) 1. + 1e-6);

    r_max = 1;
    do {
        r_max *= 2;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
        /* NOTE: This assumes v==1 uses CMP, and that CMP latency is 1 */
        for (long r = 0; r < r_max; r++)
            v == 1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
        time = (double)(ts_end.tv_sec - ts_start.tv_sec)
            + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    } while (time < MINTIME);
    freq = (double) r_max / time;


    /* vaddps */
    reg = _mm256_set1_ps(1.f);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_add_ps(reg, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    fprintf(stderr, "%i\n", (int) round(time * freq / r_max));


    /* vmulps */
    reg = _mm256_set1_ps(1.f);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_mul_ps(reg, mul0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    fprintf(stderr, "%i\n", (int) round(time * freq / r_max));


    /* vfmaddps */
    reg = _mm256_set1_ps(1.f);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_fmadd_ps(reg, mul0, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    fprintf(stderr, "%i\n", (int) round(time * freq / r_max));


    /* vsqrtps */
    reg = _mm256_set1_ps(1.f + 1e-6f);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_sqrt_ps(reg);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    fprintf(stderr, "%i\n", (int) round(time * freq / r_max));


    /* vaddps avx512 */
#ifdef USE_AVX512
    __m512 reg512;
    volatile __m512 out512;

    const __m512 add512 = _mm512_set1_ps((float) 1e-6);
    const __m512 mul512 = _mm512_set1_ps((float) 1. + 1e-6);

    reg512 = _mm512_set1_ps(1.f);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        //reg512 = _mm512_add_ps(reg512, mul512, add512);
        reg512 = _mm512_add_ps(reg512, add512);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out512 = reg512;
    fprintf(stderr, "%i\n", (int) round(time * freq / r_max));
#endif

    return 0;
}
