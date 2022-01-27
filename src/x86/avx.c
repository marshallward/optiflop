#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "roof.h"
#include "avx.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPD_LATENCY 3
#define VMULPD_LATENCY 4

/* Internal functions */
static double sum_avx(__m256d);


void avx_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(double);   // Values per SIMD register
    const int n_reg = VADDPD_LATENCY;       // Number of loop-unrolled stages
    const __m256d add0 = _mm256_set1_pd(1e-6);
    __m256d reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm256_set1_pd((double) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #pragma unroll(n_reg)
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm256_add_pd(reg[j], add0);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the results as volatile. */

    for (int j = 0; j < n_reg; j++)
        reg[0] = _mm256_add_pd(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mul(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(double);   // Values per SIMD register
    const int n_reg = VMULPD_LATENCY;     // Number of loop-unrolled stages
    const __m256d mul0 = _mm256_set1_pd(1. + 1e-6);
    __m256d reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm256_set1_pd((double) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #pragma unroll(n_reg)
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm256_mul_pd(reg[j], mul0);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the results as volatile. */

    for (int j = 0; j < n_reg; j++)
        reg[0] = _mm256_add_pd(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(double);  // Values per SIMD register
    const int n_reg = VMULPD_LATENCY;     // Number of loop-unrolled stages
    const __m256d add0 = _mm256_set1_pd(1e-6);
    const __m256d mul0 = _mm256_set1_pd(1. + 1e-6);
    __m256d reg1[n_reg];
    __m256d reg2[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    t = args->timer;

    for (j = 0; j < n_reg; j++) {
        reg1[j] = _mm256_set1_pd((double) j);
        reg2[j] = _mm256_set1_pd((double) j);
    }

    /* Add over registers r0-r4, multiply over r5-r9, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            #pragma unroll(n_reg)
            for (int j = 0; j < n_reg; j++) {
                reg1[j] = _mm256_add_pd(reg1[j], add0);
                reg2[j] = _mm256_mul_pd(reg2[j], mul0);
            }
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the result as volatile. */

    for (int j = 0; j < n_reg; j++) {
        reg1[0] = _mm256_add_pd(reg1[0], reg1[j]);
        reg1[0] = _mm256_add_pd(reg1[0], reg2[j]);
    }
    result = sum_avx(reg1[0]);

    /* Thread output */
    args->runtime = runtime;
    args->flops = r_max * n_avx * (2 * n_reg) / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


double sum_avx(__m256d x) {
    const int n_avx = 32 / sizeof(double);
    union vec {
        __m256d reg;
        double val[n_avx];
    } v;
    double result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_avx; i++)
        result += v.val[i];

    return result;
}
