#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "roof.h"
#include "avx.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 3

/* Internal functions */
static SIMDTYPE sum_avx(__m256);


void avx_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_avx = 32 / sizeof(SIMDTYPE);   // Values per SIMD register
    const int n_rolls = VADDPS_LATENCY;     // Number of loop-unrolled stages
    const __m256 add0 = _mm256_set1_ps((SIMDTYPE) 1e-6);
    __m256 reg[n_rolls];

    long r, r_max;
    int j;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile SIMDTYPE result __attribute__((unused));

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (j = 0; j < n_rolls; j++)
        reg[j] = _mm256_set1_ps((SIMDTYPE) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_rolls)
            #endif
            for (j = 0; j < n_rolls; j++)
                reg[j] = _mm256_add_ps(reg[j], add0);
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

    for (j = 0; j < n_rolls; j++)
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_rolls / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mul(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_avx = 32 / sizeof(SIMDTYPE);   // Values per SIMD register
    const int n_rolls = VMULPS_LATENCY;     // Number of loop-unrolled stages
    const __m256 mul0 = _mm256_set1_ps((SIMDTYPE) 1. + 1e-6);
    __m256 reg[n_rolls];

    long r, r_max;
    int j;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile SIMDTYPE result __attribute__((unused));

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (j = 0; j < n_rolls; j++)
        reg[j] = _mm256_set1_ps((SIMDTYPE) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_rolls)
            #endif
            for (j = 0; j < n_rolls; j++)
                reg[j] = _mm256_mul_ps(reg[j], mul0);
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

    for (j = 0; j < n_rolls; j++)
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_rolls / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_avx = 32 / sizeof(SIMDTYPE);  // Values per SIMD register
    const int n_rolls = VMULPS_LATENCY;     // Number of loop-unrolled stages
    const __m256 add0 = _mm256_set1_ps((SIMDTYPE) 1e-6);
    const __m256 mul0 = _mm256_set1_ps((SIMDTYPE) (1. + 1e-6));
    __m256 r[2 * n_rolls];  // Concurrency uses 2x registers

    // Declare as volatile to prevent removal during optimisation
    volatile SIMDTYPE result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (j = 0; j < n_rolls; j++) {
        r[j] = _mm256_set1_ps((SIMDTYPE) j);
        r[j + n_rolls] = _mm256_set1_ps((SIMDTYPE) j);
    }

    /* Add over registers r0-r4, multiply over r5-r9, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (i = 0; i < r_max; i++) {
            #ifdef __ICC
            #pragma unroll
            #endif
            for (j = 0; j < n_rolls; j++) {
                r[j] = _mm256_add_ps(r[j], add0);
                r[j + n_rolls] = _mm256_mul_ps(r[j + n_rolls], mul0);
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

    for (j = 0; j < 2 * n_rolls; j++)
        r[0] = _mm256_add_ps(r[0], r[j]);
    result = sum_avx(r[0]);

    // FLOPs: iterations * 8 FLOPs/reg * n regs/iter
    flops = r_max * n_avx * (2 * n_rolls) / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


SIMDTYPE sum_avx(__m256 x) {
    const int n_avx = 32 / sizeof(SIMDTYPE);
    union vec {
        __m256 reg;
        SIMDTYPE val[n_avx];
    } v;
    SIMDTYPE result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_avx; i++)
        result += v.val[i];

    return result;
}
