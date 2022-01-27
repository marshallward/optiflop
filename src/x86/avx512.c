#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "avx512.h"
#include "roof.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VFMAPS_LATENCY 16
#define VADDPS_LATENCY 16

/* Headers */
static double avx512_sum(__m512);


void avx512_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx512 = 64 / sizeof(double);
    const int n_reg = VADDPS_LATENCY;
    const __m512 add0 = _mm512_set1_ps((double) 1e-6);
    __m512 reg[n_reg];

    long r_max;
    double runtime, flops;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile double result;

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm512_set1_ps((double) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            #pragma unroll(n_reg)
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm512_add_ps(reg[j], add0);
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

    for (j = 0; j < n_reg; j++)
        reg[0] = _mm512_add_ps(reg[0], reg[j]);
    result = avx512_sum(reg[0]);

    /* (iter) * (16 instr / reg) * (1 flops / instr) * (n_reg reg / iter) */
    flops = r_max * n_avx512 * n_reg / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx512_fma(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_avx512 = 64 / sizeof(double);
    const int n_reg = VFMAPS_LATENCY;
    const __m512 add0 = _mm512_set1_ps((double) 1e-6);
    const __m512 mul0 = _mm512_set1_ps((double) (1. + 1e-6));
    __m512 reg[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result;

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg[j] = _mm512_set1_ps((double) j);
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
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm512_fmadd_ps(reg[j], mul0, add0);
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
        reg[0] = _mm512_add_ps(reg[0], reg[j]);
    result = avx512_sum(reg[0]);

    flops = r_max * (2 * n_avx512) * n_reg / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


double avx512_sum(__m512 x) {
    const int n_avx512 = 64 / sizeof(double);
    union vec {
        __m512 reg;
        double val[n_avx512];
    } v;
    double result = 0;

    v.reg = x;
    for (int i = 0; i < n_avx512; i++)
        result += v.val[i];

    return result;
}
