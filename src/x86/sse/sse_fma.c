#include <immintrin.h>  /* __m128, _m128_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "sse.h"
#include "bench.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 5

/* Headers */
float reduce_sse_fma(__m128);


/* Sequential SSE FMA */
void * sse_fma(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    const int n_sse = VMULPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((float) 1e-6);
    const __m128 mul0 = _mm_set1_ps((float) (1. + 1e-6));
    __m128 r[n_sse];

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct thread_args *) args_in;

    t = stopwatch_create(args->timer_type);

    for (j = 0; j < n_sse; j++) {
        r[j] = _mm_set1_ps((float) j);
    }

    runtime_flag = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < r_max; i++) {
            #ifdef __ICC
            #pragma unroll
            #endif
            for (j = 0; j < n_sse; j++) {
                r[j] = _mm_fmadd_ps(r[j], add0, mul0);
            }
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the result as volatile. */

    for (j = 0; j < n_sse; j++)
        r[0] = _mm_add_ps(r[0], r[j]);
    result = reduce_sse_fma(r[0]);

    /* (iterations) * (8 flops / register) * (n_sse registers / iteration) */
    flops = r_max * 8 * n_sse / runtime;

    /* Cleanup */
    t->destroy(t);

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}


/* Concurrent SSE FMA */
void * sse_fmac(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    const int n_sse = VMULPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((float) 1e-6);
    const __m128 mul0 = _mm_set1_ps((float) (1. + 1e-6));
    __m128 r[2 * n_sse];

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct thread_args *) args_in;

    t = stopwatch_create(args->timer_type);

    for (j = 0; j < n_sse; j++) {
        r[j] = _mm_set1_ps((float) j);
        r[j + n_sse] = _mm_set1_ps((float) j);
    }

    /* Run independent FMAs concurrently on the first and second halves of r */

    runtime_flag = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < r_max; i++) {
            #ifdef __ICC
            #pragma unroll
            #endif
            for (j = 0; j < n_sse; j++) {
                r[j] = _mm_fmadd_ps(r[j], add0, mul0);
                r[j + n_sse] = _mm_fmadd_ps(r[j + n_sse], add0, mul0);
            }
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the result as volatile. */

    for (j = 0; j < 2 * n_sse; j++)
        r[0] = _mm_add_ps(r[0], r[j]);
    result = reduce_sse_fma(r[0]);

    /* (iterations) * (8 flops / register) * (n_sse registers / iteration) */
    flops = r_max * 8 * (2 * n_sse) / runtime;

    /* Cleanup */
    t->destroy(t);

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}


/* TODO: Remove; this is identical to reduce_sse! */
float reduce_sse_fma(__m128 x) {
    union vec {
        __m128 reg;
        float val[4];
    } v;
    float result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < 4; i++)
        result += v.val[i];

    return result;
}
