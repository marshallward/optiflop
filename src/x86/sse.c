#include <immintrin.h>  /* __m128, _mm* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "sse.h"
#include "roof.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define ADDPD_LATENCY 3

/* Headers */
static double sum_sse(__m128d);


void sse_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    enum { n_sse = 16 / sizeof(double) };
    enum { n_reg = ADDPD_LATENCY };
    const __m128d add0 = _mm_set1_pd(1e-6);
    __m128d reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile double result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm_set1_pd((double) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm_add_pd(reg[j], add0);
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
        reg[0] = _mm_add_pd(reg[0], reg[j]);
    result = sum_sse(reg[0]);

    args->runtime = runtime;
    args->flops = r_max * n_sse * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


double sum_sse(__m128d x) {
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
