#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "roof.h"
#include "avx.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 3


void avx_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_simd = 32 / sizeof(SIMDTYPE);   // Values per SIMD register
    const int n_unroll = VADDPS_LATENCY;        // Loop unrolls
    const int n_reg = n_simd * n_unroll;        // Values in registers
    SIMDTYPE reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    /* Declare as volatile to prevent removal during optimization */
    volatile SIMDTYPE result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = (SIMDTYPE) j;

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++)
                reg[j] += (SIMDTYPE) 1e-6;
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

    /* Save the result in order to prevent removal during optimization. */
    result = (SIMDTYPE) 0.;
    for (int j = 0; j < n_reg; j++)
        result += reg[j];

    args->runtime = runtime;
    args->flops = r_max * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mul(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_simd = 32 / sizeof(SIMDTYPE);   // Values per SIMD register
    const int n_unroll = VMULPS_LATENCY;        // # of loop unrolls
    const int n_reg = n_simd * n_unroll;        // # of values in registers
    SIMDTYPE reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    /* Declare as volatile to prevent removal during optimization */
    volatile SIMDTYPE result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = (SIMDTYPE) j;

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++)
                reg[j] += (SIMDTYPE) 1e-6;
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

    /* Save the result in order to prevent removal during optimization. */
    result = (SIMDTYPE) 0.;
    for (int j = 0; j < n_reg; j++)
        result += reg[j];

    args->runtime = runtime;
    args->flops = r_max * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_simd = 32 / sizeof(SIMDTYPE);   // Values per SIMD register
    const int n_unroll = VADDPS_LATENCY;        // # of loop unrolls
    const int n_reg = n_simd * n_unroll;    // # of values in registers
    SIMDTYPE reg1[n_reg], reg2[n_reg];

    /* Declare as volatile to prevent removal during optimization */
    volatile SIMDTYPE result __attribute__((unused));

    long r_max;
    double runtime;
    Stopwatch *t;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg1[j] = (SIMDTYPE) j;
        reg2[j] = (SIMDTYPE) j;
    }

    /* Concurrently add the first register and multiply the second register */
    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++) {
                reg1[j] += (SIMDTYPE) 1e-6;
                reg2[j] *= (SIMDTYPE) (1. + 1e-6);
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

    /* Save the result in order to prevent removal during optimization. */
    result = (SIMDTYPE) 0.;
    for (int j = 0; j < n_reg; j++) {
        result += reg1[j];
        result += reg2[j];
    }

    /* Thread output */
    args->runtime = runtime;
    args->flops = r_max * (2 * n_reg) / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
