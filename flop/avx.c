/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <time.h>       /* timespec, clock_gettime */

#include "timer.h"

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;
const double TEST_MUL_MUL = 1.4142135623730950488;
const double TEST_MUL_DIV = 0.70710678118654752440;

const uint64_t N = 1000000000;

/* Headers */
double avx_add();
double avx_mac();
float reduce_AVX(__m256);


int main(int argc, char *argv[])
{
    float result;
    double runtime;

    runtime = avx_add();

    printf("avx_add\n");
    printf("-------\n");
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (8 registers / iteration) */
    printf("gflops: %.12f\n", N * 8 * 8 / (runtime * 1e9));

    printf("\n");

    runtime = avx_mac();

    printf("avx_mac\n");
    printf("-------\n");
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (8 registers / iteration) */
    printf("gflops: %.12f\n", N * 8 * 48 / (runtime * 1e9));

    return 0;
}


//void avx_add(float *result, double *runtime)
double avx_add()
{
    #pragma omp parallel
    {
        __m256 r[4];

        const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
        const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);

        // Declare as volatile to prevent removal during optimisation
        volatile float result;
        double runtime;

        uint64_t i, j;
        Timer *t;

        t = mtimer_create(TIMER_TSC);

        /* Select 4 numbers such that (r + a) - b != r (e.g. not 1.1f or 1.4f).
         * Some compiler optimisers (gcc) will remove the operations.
         */
        r[0] = _mm256_set1_ps(1.0f);
        r[1] = _mm256_set1_ps(1.2f);
        r[2] = _mm256_set1_ps(1.3f);
        r[3] = _mm256_set1_ps(1.5f);

        /* Add and subtract two nearly-equal double-precision numbers */

        /* Warmup */
        for (i = 0; i < N; i++) {
            for (j = 0; j < 4; j++)
                r[j] = _mm256_add_ps(r[j], add0);

            for (j = 0; j < 4; j++)
                r[j] = _mm256_sub_ps(r[j], sub0);
        }

        t->start(t);
        for (i = 0; i < N; i++) {
            for (j = 0; j < 4; j++)
                r[j] = _mm256_add_ps(r[j], add0);

            for (j = 0; j < 4; j++)
                r[j] = _mm256_sub_ps(r[j], sub0);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* In order to prevent removal of the prior loop by optimisers,
         * sum the register values and print the result. */

        /* Binomial reduction sum */
        r[0] = _mm256_add_ps(r[0], r[2]);
        r[1] = _mm256_add_ps(r[1], r[3]);
        r[0] = _mm256_add_ps(r[0], r[1]);

        /* Sum of AVX registers */
        result = reduce_AVX(r[0]);

        return runtime;
    }
}


double avx_mac()
{
    __m256 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB;

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);
    const __m256 mul0 = _mm256_set1_ps((float)TEST_MUL_MUL);
    const __m256 mul1 = _mm256_set1_ps((float)TEST_MUL_DIV);

    // Declare as volatile to prevent removal during optimisation
    volatile float result;
    double runtime;

    int i;
    Timer *t;

    t = mtimer_create(TIMER_TSC);

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

    /* Add over registers r0-r5, multiply over r6-rB
     * Rely on pipelining and latency difference (3 vs 5 cycles) for 2x FLOPs
     */

    t->start(t);
#pragma distribute_point
    for (i = 0; i < N; i++) {
        r6 = _mm256_mul_ps(r6, mul0);
        r0 = _mm256_add_ps(r0, add0);

        r7 = _mm256_mul_ps(r7, mul0);
        r1 = _mm256_add_ps(r1, add0);

        r8 = _mm256_mul_ps(r8, mul0);
        r2 = _mm256_add_ps(r2, add0);

        r9 = _mm256_mul_ps(r9, mul0);
        r3 = _mm256_add_ps(r3, add0);

        rA = _mm256_mul_ps(rA, mul0);
        r4 = _mm256_add_ps(r4, add0);

        rB = _mm256_mul_ps(rB, mul0);
        r5 = _mm256_add_ps(r5, add0);

        r6 = _mm256_mul_ps(r6, mul1);
        r0 = _mm256_sub_ps(r0, sub0);

        r7 = _mm256_mul_ps(r7, mul1);
        r1 = _mm256_sub_ps(r1, sub0);

        r8 = _mm256_mul_ps(r8, mul1);
        r2 = _mm256_sub_ps(r2, sub0);

        r9 = _mm256_mul_ps(r9, mul1);
        r3 = _mm256_sub_ps(r3, sub0);

        rA = _mm256_mul_ps(rA, mul1);
        r4 = _mm256_sub_ps(r4, sub0);

        rB = _mm256_mul_ps(rB, mul1);
        r5 = _mm256_sub_ps(r5, sub0);

        /* repeat */
        r6 = _mm256_mul_ps(r6, mul0);
        r0 = _mm256_add_ps(r0, add0);

        r7 = _mm256_mul_ps(r7, mul0);
        r1 = _mm256_add_ps(r1, add0);

        r8 = _mm256_mul_ps(r8, mul0);
        r2 = _mm256_add_ps(r2, add0);

        r9 = _mm256_mul_ps(r9, mul0);
        r3 = _mm256_add_ps(r3, add0);

        rA = _mm256_mul_ps(rA, mul0);
        r4 = _mm256_add_ps(r4, add0);

        rB = _mm256_mul_ps(rB, mul0);
        r5 = _mm256_add_ps(r5, add0);

        r6 = _mm256_mul_ps(r6, mul1);
        r0 = _mm256_sub_ps(r0, sub0);

        r7 = _mm256_mul_ps(r7, mul1);
        r1 = _mm256_sub_ps(r1, sub0);

        r8 = _mm256_mul_ps(r8, mul1);
        r2 = _mm256_sub_ps(r2, sub0);

        r9 = _mm256_mul_ps(r9, mul1);
        r3 = _mm256_sub_ps(r3, sub0);

        rA = _mm256_mul_ps(rA, mul1);
        r4 = _mm256_sub_ps(r4, sub0);

        rB = _mm256_mul_ps(rB, mul1);
        r5 = _mm256_sub_ps(r5, sub0);
    }
    t->stop(t);
    runtime = t->runtime(t);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and print the result. */

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

    return runtime;
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
