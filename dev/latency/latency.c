#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#define MINTIME 0.1

int main(int argc, char *argv[])
{
    struct timespec ts_start, ts_end;
    double time;
    __m128d reg_sse;
    __m256d reg;
    volatile __m128d out_sse;
    volatile __m256d out;

    long r_max;
    volatile long v;
    double freq;

    const __m128d eps_sse = _mm_set1_pd(1e-6);
    const __m256d add0 = _mm256_set1_pd(1e-6);
    const __m256d mul0 = _mm256_set1_pd(1. + 1e-6);

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

    /* addps */
    reg_sse = _mm_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg_sse = _mm_add_pd(reg_sse, eps_sse);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out_sse = reg_sse;
    //fprintf(stderr, "%i\n", (int) round(time * freq / r_max));
    printf("addps: %i\n", (int) round(time * freq / r_max));


    /* vaddps */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_add_pd(reg, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    printf("vaddps: %i\n", (int) round(time * freq / r_max));


    /* vmulps */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_mul_pd(reg, mul0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    printf("vmulps: %i\n", (int) round(time * freq / r_max));


    /* vfmaddps */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_fmadd_pd(reg, mul0, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    printf("vfmaddps: %i\n", (int) round(time * freq / r_max));


    /* vsqrtps */
    reg = _mm256_set1_pd(1. + 1e-6);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_sqrt_pd(reg);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    printf("vsqrtps: %i\n", (int) round(time * freq / r_max));


    /* vaddps avx512 */
#ifdef USE_AVX512
    __m512d reg512;
    volatile __m512d out512;

    const __m512d add512 = _mm512_set1_pd(1e-6);
    const __m512d mul512 = _mm512_set1_pd(1. + 1e-6);

    reg512 = _mm512_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        //reg512 = _mm512_add_pd(reg512, mul512, add512);
        reg512 = _mm512_add_pd(reg512, add512);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out512 = reg512;
    printf("(512) vaddps: %i\n", (int) round(time * freq / r_max));
#endif

    return 0;
}
