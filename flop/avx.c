/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "timer.h"

pthread_barrier_t timer_barrier;
pthread_mutex_t runtime_mutex;
volatile int runtime_flag;

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;
const double TEST_MUL_MUL = 1.4142135623730950488;
const double TEST_MUL_DIV = 0.70710678118654752440;

/* Headers */
float reduce_AVX(__m256);

void avx_add(double *runtime, double *flops)
{
    // TODO: Stop using outputs as intermediate values
    __m256 r[4];

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);

    // Declare as volatile to prevent removal during optimisation
    volatile float result;

    uint64_t i, j;
    long niter;
    Timer *t;

    t = mtimer_create(TIMER_POSIX);

    /* Select 4 numbers such that (r + a) - b != r (e.g. not 1.1f or 1.4f).
     * Some compiler optimisers (gcc) will remove the operations.
     */
    r[0] = _mm256_set1_ps(1.0f);
    r[1] = _mm256_set1_ps(1.2f);
    r[2] = _mm256_set1_ps(1.3f);
    r[3] = _mm256_set1_ps(1.5f);

    /* Add and subtract two nearly-equal double-precision numbers */

    // XXX: Barrier will hang if some threads reach 0.5s before others
    runtime_flag = 0;
    niter = 1000;
    do {
        niter *= 2;

        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < niter; i++) {
            r[0] = _mm256_add_ps(r[0], add0);
            r[1] = _mm256_add_ps(r[1], add0);
            r[2] = _mm256_add_ps(r[2], add0);
            r[3] = _mm256_add_ps(r[3], add0);

            r[0] = _mm256_sub_ps(r[0], sub0);
            r[1] = _mm256_sub_ps(r[1], sub0);
            r[2] = _mm256_sub_ps(r[2], sub0);
            r[3] = _mm256_sub_ps(r[3], sub0);
        }
        t->stop(t);
        *runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        /* (Do I really need the mutex here?) */
        if (*runtime > 0.5) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }
        pthread_barrier_wait(&timer_barrier);
    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the results as volatile. */

    /* Binomial reduction sum */
    r[0] = _mm256_add_ps(r[0], r[2]);
    r[1] = _mm256_add_ps(r[1], r[3]);
    r[0] = _mm256_add_ps(r[0], r[1]);

    /* Sum of AVX registers */
    result = reduce_AVX(r[0]);

    *flops = niter * 8 * 8 / *runtime;
}


void avx_mac(double *runtime, double *flops)
{
    __m256 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB;

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);
    const __m256 mul0 = _mm256_set1_ps((float)TEST_MUL_MUL);
    const __m256 mul1 = _mm256_set1_ps((float)TEST_MUL_DIV);

    // Declare as volatile to prevent removal during optimisation
    volatile float result;

    uint64_t i;
    long niter;
    Timer *t;

    t = mtimer_create(TIMER_POSIX);

    /* Scatter values over AVX registers */

    /* Choose non-exact sums (r + a) - b, (r * a) / c */
    r0 = _mm256_set1_ps(1.0f);
    r1 = _mm256_set1_ps(1.2f);
    r2 = _mm256_set1_ps(1.3f);
    r3 = _mm256_set1_ps(1.5f);
    r4 = _mm256_set1_ps(1.7f);
    r5 = _mm256_set1_ps(1.8f);

    r6 = _mm256_set1_ps(1.0f);
    r7 = _mm256_set1_ps(1.3f);
    r8 = _mm256_set1_ps(1.5f);
    r9 = _mm256_set1_ps(1.8f);
    rA = _mm256_set1_ps(2.0f);
    rB = _mm256_set1_ps(2.6f);

    /* Add over registers r0-r5, multiply over r6-rB, and rely on pipelining,
     * OOO execution, and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    // XXX: Barrier will hang if some threads reach 0.5s before others
    runtime_flag = 0;
    niter = 1000;
    do {
        niter *= 2;

        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (i = 0; i < niter; i++) {
            r0 = _mm256_add_ps(r0, add0);
            r1 = _mm256_add_ps(r1, add0);
            r2 = _mm256_add_ps(r2, add0);
            r3 = _mm256_add_ps(r3, add0);
            r4 = _mm256_add_ps(r4, add0);
            r5 = _mm256_add_ps(r5, add0);

            r6 = _mm256_mul_ps(r6, mul0);
            r7 = _mm256_mul_ps(r7, mul0);
            r8 = _mm256_mul_ps(r8, mul0);
            r9 = _mm256_mul_ps(r9, mul0);
            rA = _mm256_mul_ps(rA, mul0);
            rB = _mm256_mul_ps(rB, mul0);

            r0 = _mm256_sub_ps(r0, sub0);
            r1 = _mm256_sub_ps(r1, sub0);
            r2 = _mm256_sub_ps(r2, sub0);
            r3 = _mm256_sub_ps(r3, sub0);
            r4 = _mm256_sub_ps(r4, sub0);
            r5 = _mm256_sub_ps(r5, sub0);

            r6 = _mm256_mul_ps(r6, mul1);
            r7 = _mm256_mul_ps(r7, mul1);
            r8 = _mm256_mul_ps(r8, mul1);
            r9 = _mm256_mul_ps(r9, mul1);
            rA = _mm256_mul_ps(rA, mul1);
            rB = _mm256_mul_ps(rB, mul1);
        }
        t->stop(t);
        *runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        /* (Do I really need the mutex here?) */
        if (*runtime > 0.5) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }
        pthread_barrier_wait(&timer_barrier);
    } while (!runtime_flag);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and save the result as volatile. */

    /* Binomial reduction sum */
    r0 = _mm256_add_ps(r0, r6);
    r1 = _mm256_add_ps(r1, r7);
    r2 = _mm256_add_ps(r2, r8);
    r3 = _mm256_add_ps(r3, r9);
    r4 = _mm256_add_ps(r4, rA);
    r5 = _mm256_add_ps(r5, rB);

    r0 = _mm256_add_ps(r0, r3);
    r1 = _mm256_add_ps(r1, r4);
    r2 = _mm256_add_ps(r2, r5);

    r0 = _mm256_add_ps(r0, r1);
    r0 = _mm256_add_ps(r0, r2);

    /* Sum of AVX registers */
    result = reduce_AVX(r0);

    *flops = niter * 8 * 24 / *runtime;
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
