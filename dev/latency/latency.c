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

    int addpd, vaddpd, vmulpd, vfmaddpd, vsqrtpd;

    const __m128d eps_sse = _mm_set1_pd(1e-6);
    const __m256d add0 = _mm256_set1_pd(1e-6);
    const __m256d mul0 = _mm256_set1_pd(1. + 1e-6);

    /* This is not correct; the "scalar" frequency is not necessarily the "AVX"
     * frequency. */
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
    printf("freq: %.12f\n", freq * 1e-9);
    printf("rmax: %li\n", r_max);

    /****************/

    const int nreg = 4;
    __m256d rset[nreg];
    volatile __m256d oset[nreg];

    /* vaddpd */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_add_pd(reg, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    printf("t1: %.12f\n", time*1e9);

    /* vaddpd */
    for (int i = 0; i < nreg; i++)
        rset[i] = _mm256_set1_pd((float) i);

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        for (int i = 0; i < nreg; i++) {
            rset[i] = _mm256_add_pd(rset[i], add0);
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    for (int i = 0; i < nreg; i++) {
        oset[i] = rset[i];
    }
    printf("t2: %.12f\n", time*1e9);

    //exit(0);

    /****************/

    /* addpd */
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
    addpd = (int) round(time * freq / r_max);
    printf("addpd: %i\n", addpd);


    /* vaddpd */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_add_pd(reg, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    vaddpd = (int) round(time * freq / r_max);
    printf("vaddpd: %i\n", vaddpd);


    /* vmulpd */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_mul_pd(reg, mul0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    vmulpd = (int) round(time * freq / r_max);
    printf("vmulpd: %i\n", vmulpd);


    /* vfmaddpd */
    reg = _mm256_set1_pd(1.);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_fmadd_pd(reg, mul0, add0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    vfmaddpd = (int) round(time * freq / r_max);
    printf("vfmaddpd: %i\n", vfmaddpd);


    /* vsqrtpd */
    reg = _mm256_set1_pd(1. + 1e-6);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (long r = 0; r < r_max; r++) {
        reg = _mm256_sqrt_pd(reg);
        reg = _mm256_mul_pd(reg, reg);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    time = (double)(ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    out = reg;
    vsqrtpd = (int) round(time * freq / r_max) - vmulpd;
    printf("vsqrtpd: %i\n", vsqrtpd);


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
    printf("(512) vaddpd: %i\n", (int) round(time * freq / r_max));
#endif

    return 0;
}
