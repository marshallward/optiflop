/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "avx.h"
#include "bench.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 5

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;
const double TEST_MUL_MUL = 1.4142135623730950488;
const double TEST_MUL_DIV = 0.70710678118654752440;

/* Headers */
float reduce_AVX(__m256);

void avx_add(bench_arg_t *args)
{
    const int n_avx = VADDPS_LATENCY;
    __m256 r[n_avx];

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);

    // Declare as volatile to prevent removal during optimisation
    volatile float result;

    long niter, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    t = stopwatch_create(TIMER_POSIX);

    /* Select 4 numbers such that (r + a) - b != r (e.g. not 1.1f or 1.4f).
     * Some compiler optimisers (gcc) will remove the operations.
     * The vaddps 3-cycle latency requires 3 concurrent operations
     */
    r[0] = _mm256_set1_ps(1.0f);
    r[1] = _mm256_set1_ps(1.2f);
    r[2] = _mm256_set1_ps(1.3f);

    /* Add and subtract two nearly-equal double-precision numbers */

    runtime_flag = 0;
    niter = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < niter; i++) {
            for (j = 0; j < n_avx; j++)
                r[j] = _mm256_add_ps(r[j], add0);

            for (j = 0; j < n_avx; j++)
                r[j] = _mm256_sub_ps(r[j], sub0);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        } else {
            niter *= 2;
        }

        pthread_barrier_wait(&timer_barrier);
    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the results as volatile. */

    r[0] = _mm256_add_ps(r[0], r[2]);
    r[0] = _mm256_add_ps(r[0], r[1]);
    result = reduce_AVX(r[0]);

    /* (iterations) * (8 flops / register) * (6 registers / iteration) */
    flops = niter * 8 * 6 / runtime;

    /* Cleanup */
    args->runtime = runtime;
    args->flops = flops;
    t->destroy(t);
}


void avx_mac(bench_arg_t *args)
{
    __m256 r[10];

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);
    const __m256 mul0 = _mm256_set1_ps((float)TEST_MUL_MUL);
    const __m256 mul1 = _mm256_set1_ps((float)TEST_MUL_DIV);

    // Declare as volatile to prevent removal during optimisation
    volatile float result;

    long niter, i;
    double runtime, flops;
    Stopwatch *t;

    t = stopwatch_create(TIMER_POSIX);

    /* Scatter values over AVX registers */

    /* Choose non-exact sums (r + a) - b, (r * a) / c */
    /* The vmulps 5-cycle latency requires 5 concurrent operations */
    r[0] = _mm256_set1_ps(1.0f);
    r[1] = _mm256_set1_ps(1.2f);
    r[2] = _mm256_set1_ps(1.3f);
    r[3] = _mm256_set1_ps(1.5f);
    r[4] = _mm256_set1_ps(1.7f);

    r[5] = _mm256_set1_ps(1.0f);
    r[6] = _mm256_set1_ps(1.3f);
    r[7] = _mm256_set1_ps(1.5f);
    r[8] = _mm256_set1_ps(1.8f);
    r[9] = _mm256_set1_ps(2.0f);

    /* Add over registers r0-r4, multiply over r5-r9, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    runtime_flag = 0;
    niter = 1;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < niter; i++) {
            r[0] = _mm256_add_ps(r[0], add0);
            r[1] = _mm256_add_ps(r[1], add0);
            r[2] = _mm256_add_ps(r[2], add0);
            r[3] = _mm256_add_ps(r[3], add0);
            r[4] = _mm256_add_ps(r[4], add0);

            r[5] = _mm256_mul_ps(r[5], mul0);
            r[6] = _mm256_mul_ps(r[6], mul0);
            r[7] = _mm256_mul_ps(r[7], mul0);
            r[8] = _mm256_mul_ps(r[8], mul0);
            r[9] = _mm256_mul_ps(r[9], mul0);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        /* (Do I really need the mutex here?) */
        if (runtime > args->min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        } else {
            niter *= 2;
        }

        pthread_barrier_wait(&timer_barrier);
    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the result as volatile. */

    /* Binomial reduction sum */
    for (i = 0; i < 5; i++)
        r[i] = _mm256_add_ps(r[i], r[i + 5]);

    for (i = 0; i < 2; i++)
        r[i] = _mm256_add_ps(r[i], r[i + 2]);
    r[0] = _mm256_add_ps(r[0], r[4]);

    r[0] = _mm256_add_ps(r[0], r[1]);

    /* Sum of AVX registers */
    result = reduce_AVX(r[0]);

    /* (iterations) * (8 flops / register) * (10 registers / iteration) */
    flops = niter * 8 * 10 / runtime;

    /* Cleanup */
    args->runtime = runtime;
    args->flops = flops;
    t->destroy(t);
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
