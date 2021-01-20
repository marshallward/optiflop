#include <immintrin.h>  /* __m128, _mm* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "sse.h"
#include "bench.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 5

/* Headers */
float reduce_sse(__m128);


void sse_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    const int n_sse = VADDPS_LATENCY;
    const __m128 add0 = _mm_set1_ps((float) 1e-6);
    __m128 reg[n_sse];

    long r, r_max;
    int j;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    /* Read inputs */
    args = (struct roof_args *) args_in;

    t = args->timer;

    for (j = 0; j < n_sse; j++)
        reg[j] = _mm_set1_ps((float) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_sse)
            #endif
            for (j = 0; j < n_sse; j++)
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

    for (j = 0; j < n_sse; j++)
        reg[0] = _mm_add_ps(reg[0], reg[j]);
    result = reduce_sse(reg[0]);

    args->runtime = runtime;
    args->flops = r_max * 4 * n_sse / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


float reduce_sse(__m128 x) {
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
