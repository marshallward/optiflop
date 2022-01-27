#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "roof.h"
#include "avx.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 3

/* Internal functions */
static float sum_avx(__m256);


void avx_add(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(float);   // Values per SIMD register
    const int n_reg = VADDPS_LATENCY;       // Number of loop-unrolled stages
    const __m256 add0 = _mm256_set1_ps((float) 1e-6);
    __m256 reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm256_set1_ps((float) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_reg)
            #endif
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm256_add_ps(reg[j], add0);
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
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mul(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(float);   // Values per SIMD register
    const int n_reg = VMULPS_LATENCY;     // Number of loop-unrolled stages
    const __m256 mul0 = _mm256_set1_ps((float) 1. + 1e-6);
    __m256 reg[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++)
        reg[j] = _mm256_set1_ps((float) j);

    *(args->runtime_flag) = 0;
    r_max = 1;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_reg)
            #endif
            for (int j = 0; j < n_reg; j++)
                reg[j] = _mm256_mul_ps(reg[j], mul0);
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
        reg[0] = _mm256_add_ps(reg[0], reg[j]);
    result = sum_avx(reg[0]);

    // FLOPs: iterations * N FLOPs/reg * n regs/iter
    args->runtime = runtime;
    args->flops = r_max * n_avx * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mac(void *args_in)
{
    /* Thread input */
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    const int n_avx = 32 / sizeof(float);  // Values per SIMD register
    const int n_reg = VMULPS_LATENCY;     // Number of loop-unrolled stages
    const __m256 add0 = _mm256_set1_ps((float) 1e-6);
    const __m256 mul0 = _mm256_set1_ps((float) (1. + 1e-6));
    __m256 reg1[n_reg];
    __m256 reg2[n_reg];

    // Declare as volatile to prevent removal during optimisation
    volatile float result __attribute__((unused));

    long r_max, i;
    int j;
    double runtime, flops;
    Stopwatch *t;

    t = args->timer;

    for (j = 0; j < n_reg; j++) {
        reg1[j] = _mm256_set1_ps((float) j);
        reg2[j] = _mm256_set1_ps((float) j);
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
            #ifdef __ICC
            #pragma unroll
            #endif
            for (int j = 0; j < n_reg; j++) {
                reg1[j] = _mm256_add_ps(reg1[j], add0);
                reg2[j] = _mm256_mul_ps(reg2[j], mul0);
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

    for (int j = 0; j < n_reg; j++) {
        reg1[0] = _mm256_add_ps(reg1[0], reg1[j]);
        reg1[0] = _mm256_add_ps(reg1[0], reg2[j]);
    }
    result = sum_avx(reg1[0]);

    /* Thread output */
    args->runtime = runtime;
    args->flops = r_max * n_avx * (2 * n_reg) / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


float sum_avx(__m256 x) {
    const int n_avx = 32 / sizeof(float);
    union vec {
        __m256 reg;
        float val[n_avx];
    } v;
    float result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_avx; i++)
        result += v.val[i];

    return result;
}
