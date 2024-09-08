#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "avx.h"
#include "roof.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VFMAPD_LATENCY 4

/* Headers */
static double sum_avx(__m256);


/* Sequential AVX FMA */
void avx_fma(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    enum { n_avx = 32 / sizeof(double) };
    enum { n_reg = VFMAPD_LATENCY };
    const __m256 add0 = _mm256_set1_ps(1e-6);
    const __m256 mul0 = _mm256_set1_ps(1. + 1e-6);
    __m256 reg[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    long r_max;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg[j] = _mm256_set1_ps((double) j);
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
            for (int j = 0; j < n_reg; j++) {
                reg[j] = _mm256_fmadd_ps(reg[j], add0, mul0);
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

    for (int j = 0; j < n_reg; j++)
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    /* (iterations) * (2 * 8 flops / register) * (n_reg registers / iteration) */
    flops = r_max * 2 * n_avx * n_reg / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


/* Concurrent AVX FMA */
void avx_fmac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    enum { n_avx = 32 / sizeof(double) };
    enum { n_reg = VFMAPD_LATENCY };

    const __m256 add0 = _mm256_set1_ps(1e-6);
    const __m256 mul0 = _mm256_set1_ps(1. + 1e-6);
    __m256 reg[2 * n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    long r_max;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg[j] = _mm256_set1_ps((double) j);
        reg[j + n_reg] = _mm256_set1_ps((double) j);
    }

    /* Add over registers r0-r4, multiply over r5-r9, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (int i = 0; i < r_max; i++) {
            for (int j = 0; j < n_reg; j++) {
                reg[j] = _mm256_fmadd_ps(reg[j], add0, mul0);
                reg[j + n_reg] = _mm256_fmadd_ps(reg[j + n_reg], add0, mul0);
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

    for (int j = 0; j < 2 * n_reg; j++)
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    /* (iterations) * (16 flops / register) * (n_reg registers / iteration) */
    flops = r_max * 2 * n_avx * (2 * n_reg) / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


double sum_avx(__m256 x) {
    enum { n_avx = 32 / sizeof(double) };
    union vec {
        __m256 reg;
        double val[n_avx];
    } v;
    double result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_avx; i++)
        result += v.val[i];

    return result;
}
