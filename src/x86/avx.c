#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "avx.h"
#include "bench.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 5

/* Headers */
float reduce_AVX(__m256);


void * avx_add(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    const int n_avx = VADDPS_LATENCY;
    const __m256 add0 = _mm256_set1_ps((float) 1e-6);
    __m256 reg[n_avx];

    long r, r_max;
    int j;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    /* Read inputs */
    args = (struct thread_args *) args_in;

    t = stopwatch_create(args->timer_type);

    for (j = 0; j < n_avx; j++)
        reg[j] = _mm256_set1_ps((float) j);

    runtime_flag = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #pragma unroll(n_avx)
            for (j = 0; j < n_avx; j++)
                reg[j] = _mm256_add_ps(reg[j], add0);
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
     * sum the register values and save the results as volatile. */

    for (j = 0; j < n_avx; j++)
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = reduce_AVX(reg[0]);

    args->runtime = runtime;
    args->flops = r_max * 8 * n_avx / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;

    /* Cleanup */
    t->destroy(t);

    pthread_exit(NULL);
}


void * avx_mac(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    const int n_avx = VMULPS_LATENCY;
    const __m256 add0 = _mm256_set1_ps((float) 1e-6);
    const __m256 mul0 = _mm256_set1_ps((float) (1. + 1e-6));
    __m256 r[2 * n_avx];

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct thread_args *) args_in;

    t = stopwatch_create(args->timer_type);

    for (j = 0; j < n_avx; j++) {
        r[j] = _mm256_set1_ps((float) j);
        r[j + n_avx] = _mm256_set1_ps((float) j);
    }

    /* Add over registers r0-r4, multiply over r5-r9, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    runtime_flag = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < r_max; i++) {
            #pragma unroll
            for (j = 0; j < n_avx; j++) {
                r[j] = _mm256_add_ps(r[j], add0);
                r[j + n_avx] = _mm256_mul_ps(r[j + n_avx], mul0);
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

    for (j = 0; j < 2 * n_avx; j++)
        r[0] = _mm256_add_ps(r[0], r[j]);
    result = reduce_AVX(r[0]);

    /* (iterations) * (8 flops / register) * (2*n_avx registers / iteration) */
    flops = r_max * 8 * (2 * n_avx) / runtime;

    /* Cleanup */
    t->destroy(t);

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}


float reduce_AVX(__m256 x) {
    union vec {
        __m256 reg;
        float val[8];
    } v;
    float result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < 8; i++)
        result += v.val[i];

    return result;
}
