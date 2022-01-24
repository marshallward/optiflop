#include <immintrin.h>  /* __m128, _m128_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "sse.h"
#include "roof.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 5

/* Headers */
static SIMDTYPE sse_sum(__m128);


/* Sequential SSE FMA */
void sse_fma(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_sse = 16 / sizeof(SIMDTYPE);
    const int n_rolls = VMULPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((SIMDTYPE) 1e-6);
    const __m128 mul0 = _mm_set1_ps((SIMDTYPE) (1. + 1e-6));
    __m128 r[n_rolls];

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
        r[j] = _mm_set1_ps((SIMDTYPE) j);
    }

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
                r[j] = _mm_fmadd_ps(r[j], add0, mul0);
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

    for (j = 0; j < n_rolls; j++)
        r[0] = _mm_add_ps(r[0], r[j]);
    result = sse_sum(r[0]);

    flops = r_max * (2 * n_sse) * n_rolls / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


/* Concurrent SSE FMA */
void sse_fmac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_sse = 16 / sizeof(SIMDTYPE);
    const int n_rolls = VMULPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((SIMDTYPE) 1e-6);
    const __m128 mul0 = _mm_set1_ps((SIMDTYPE) (1. + 1e-6));
    __m128 r[2 * n_rolls];

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
        r[j] = _mm_set1_ps((SIMDTYPE) j);
        r[j + n_rolls] = _mm_set1_ps((SIMDTYPE) j);
    }

    /* Run independent FMAs concurrently on the first and second halves of r */

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
                r[j] = _mm_fmadd_ps(r[j], add0, mul0);
                r[j + n_rolls] = _mm_fmadd_ps(r[j + n_rolls], add0, mul0);
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
        r[0] = _mm_add_ps(r[0], r[j]);
    result = sse_sum(r[0]);

    flops = r_max * (2 * n_sse) * (2 * n_rolls) / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


SIMDTYPE sse_sum(__m128 x) {
    const int n_sse = 16 / sizeof(SIMDTYPE);
    union vec {
        __m128 reg;
        SIMDTYPE val[n_sse];
    } v;
    SIMDTYPE result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_sse; i++)
        result += v.val[i];

    return result;
}
