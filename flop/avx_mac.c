/* FLOP test (based heavily on Alex Yee source) */
/* Multiply + Add */

#include <immintrin.h> /* __m256, _m256_* */
#include <stdint.h> /* uint64_t */
#include <stdio.h> /* printf */
#include <time.h> /* timespec, clock_gettime */

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;
const double TEST_MUL_MUL = 1.4142135623730950488;
const double TEST_MUL_DIV = 0.70710678118654752440;

const uint64_t N = 100000000;

#define USE_RDTSC
//const double CPUFREQ = 2.593966925e9;     // My desktop?
const double CPUFREQ = 2.601e9;             // Raijin (?)

/* Headers */
float reduce_AVX(__m256);

int main(int argc, char *argv[])
{
    __m256 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB;
    
    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);
    const __m256 mul0 = _mm256_set1_ps((float)TEST_MUL_MUL);
    const __m256 mul1 = _mm256_set1_ps((float)TEST_MUL_DIV);

#ifdef USE_RDTSC
    uint64_t rax0, rdx0, rax1, rdx1;
    uint64_t t0, t1;
#else
    struct timespec ts_start, ts_end;
#endif
    double runtime;
    int i;
    float result;

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

#ifdef USE_RDTSC
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (rax0), "=r" (rdx0)
        :: "%rax", "%rbx", "%rcx", "%rdx");
        // "movq %%rdx, %0\n"
        // "movq %%rax, %1\n"
        // : "=r" (rdx), "=r" (rax) :: "%rax", "%rdx");
#else
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
#endif

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
#ifdef USE_RDTSC
    __asm__ __volatile__ (
            "rdtscp\n"
            "movq %%rax, %0\n"
            "movq %%rdx, %1\n"
            "cpuid\n"
            : "=r" (rax1), "=r" (rdx1)
            :: "%rax", "%rbx", "%rcx", "%rdx" );

    t0 = (rdx0 << 32) | rax0;
    t1 = (rdx1 << 32) | rax1;
    runtime = (t1 - t0) / CPUFREQ;
    printf("TSC count: %lu\n", t1 - t0);
#else
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
             + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
#endif

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

    printf("result: %f\n", result);
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (48 registers / iteration) */
    //printf("gflops: %.12f\n", N * 8 * 48 / (runtime * 1e9));
    printf("gflops: %.12f\n", N * 8 * 48 / (runtime * 1e9));

    return 0;
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
