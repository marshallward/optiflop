#include <immintrin.h>  /* __m128, _mm* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "sse.h"
#include "roof.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define ADDPS_LATENCY 3

/* Headers */
static float sum_sse(__m128);


void sse_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_sse = 16 / sizeof(float);
    const int n_rolls = ADDPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((float) 1e-6);
    __m128 reg[n_rolls];

    long r, r_max;
    int j;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (j = 0; j < n_rolls; j++)
        reg[j] = _mm_set1_ps((float) j);

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
                reg[j] = _mm_add_ps(reg[j], add0);
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
        reg[0] = _mm_add_ps(reg[0], reg[j]);
    result = sum_sse(reg[0]);

    args->runtime = runtime;
    args->flops = r_max * n_sse * n_rolls / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


float sum_sse(__m128 x) {
    const int n_sse = 16 / sizeof(float);
    union vec {
        __m128 reg;
        float val[n_sse];
    } v;
    float result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < sizeof(float); i++)
        result += v.val[i];

    return result;
}
