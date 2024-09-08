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
static double sse_sum(__m128d);


/* Sequential SSE FMA */
void sse_fma(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    enum { n_sse = 16 / sizeof(double) };
    enum { n_reg = VMULPS_LATENCY };
    const __m128d add0 = _mm_set1_pd(1e-6);
    const __m128d mul0 = _mm_set1_pd(1. + 1e-6);
    __m128d reg[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    long r_max;
    double runtime, flops;
    Stopwatch *t;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg[j] = _mm_set1_pd((double) j);
    }

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (int r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++) {
                reg[j] = _mm_fmadd_pd(reg[j], add0, mul0);
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
        reg[0] = _mm_add_pd(reg[0], reg[j]);
    result = sse_sum(reg[0]);

    flops = r_max * (2 * n_sse) * n_reg / runtime;

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

    enum { n_sse = 16 / sizeof(double) };
    enum { n_reg = VMULPS_LATENCY };
    const __m128d add0 = _mm_set1_pd(1e-6);
    const __m128d mul0 = _mm_set1_pd(1. + 1e-6);
    //__m128d reg[2 * n_reg];
    __m128d reg1[n_reg];
    __m128d reg2[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    long r_max;
    double runtime, flops;
    Stopwatch *t;

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        //reg[j] = _mm_set1_pd((double) j);
        //reg[j + n_reg] = _mm_set1_pd((double) j);
        reg1[j] = _mm_set1_pd((double) j);
        reg2[j] = _mm_set1_pd((double) j);
    }

    /* Run independent FMAs concurrently on the first and second halves of r */

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++) {
                //reg[j] = _mm_fmadd_pd(reg[j], add0, mul0);
                //reg[j + n_reg] = _mm_fmadd_pd(reg[j + n_reg], add0, mul0);
                reg1[j] = _mm_fmadd_pd(reg1[j], add0, mul0);
                reg2[j] = _mm_fmadd_pd(reg2[j], add0, mul0);
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

    //for (j = 0; j < 2 * n_reg; j++)
    //    reg[0] = _mm_add_pd(reg[0], reg[j]);
    for (int j = 0; j < n_reg; j++) {
        reg1[0] = _mm_add_pd(reg1[0], reg1[j]);
        reg1[0] = _mm_add_pd(reg1[0], reg2[j]);
    }
    result = sse_sum(reg1[0]);

    flops = r_max * (2 * n_sse) * (2 * n_reg) / runtime;

    /* Thread output */
    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


double sse_sum(__m128d x) {
    enum { n_sse = 16 / sizeof(double) };
    union vec {
        __m128d reg;
        double val[n_sse];
    } v;
    double result = 0;

    v.reg = x;
    for (int i = 0; i < n_sse; i++)
        result += v.val[i];

    return result;
}
