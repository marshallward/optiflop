/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h> /* __m256, _m256_* */
#include <stdint.h> /* uint64_t */
#include <stdio.h> /* printf */
#include <time.h> /* timespec, clock_gettime */

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;

const uint64_t N = 100000000;

/* Headers */
float reduce_AVX(__m256);

int main(int argc, char *argv[])
{
    //__m256 r0, r1, r2, r3;
    __m256 r[4];

    const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
    const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);

    struct timespec ts_start, ts_end;
    double runtime;
    int i, j;
    float result;

    /* Select 4 numbers such that (r + a) - b != r (e.g. not 1.1f or 1.4f).
     * Some compiler optimisers (gcc) will remove the operations.
     */
    r[0] = _mm256_set1_ps(1.0f);
    r[1] = _mm256_set1_ps(1.2f);
    r[2] = _mm256_set1_ps(1.3f);
    r[3] = _mm256_set1_ps(1.5f);

    /* Add and subtract two nearly-equal double-precision numbers */
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (i = 0; i < N; i++) {
        /* Manual loop unroll */
        /*
        r[0] = _mm256_add_ps(r[0], add0);
        r[1] = _mm256_add_ps(r[1], add0);
        r[2] = _mm256_add_ps(r[2], add0);
        r[3] = _mm256_add_ps(r[3], add0);
        r[0] = _mm256_sub_ps(r[0], sub0);
        r[1] = _mm256_sub_ps(r[1], sub0);
        r[2] = _mm256_sub_ps(r[2], sub0);
        r[3] = _mm256_sub_ps(r[3], sub0);

        r[0] = _mm256_add_ps(r[0], add0);
        r[1] = _mm256_add_ps(r[1], add0);
        r[2] = _mm256_add_ps(r[2], add0);
        r[3] = _mm256_add_ps(r[3], add0);
        r[0] = _mm256_sub_ps(r[0], sub0);
        r[1] = _mm256_sub_ps(r[1], sub0);
        r[2] = _mm256_sub_ps(r[2], sub0);
        r[3] = _mm256_sub_ps(r[3], sub0);
        */
        for (j = 0; j < 4; j++)
            r[j] = _mm256_add_ps(r[j], add0);

        for (j = 0; j < 4; j++)
            r[j] = _mm256_sub_ps(r[j], sub0);

        for (j = 0; j < 4; j++)
            r[j] = _mm256_add_ps(r[j], add0);

        for (j = 0; j < 4; j++)
            r[j] = _mm256_sub_ps(r[j], sub0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    /* In order to prevent removal of the prior loop by optimisers,
     * sum the register values and print the result. */

    /* Binomial reduction sum */
    r[0] = _mm256_add_ps(r[0], r[2]);
    r[1] = _mm256_add_ps(r[1], r[3]);
    r[0] = _mm256_add_ps(r[0], r[1]);

    /* Sum of AVX registers */
    result = reduce_AVX(r[0]);

    runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("result: %f\n", result);
    printf("runtime: %.12f\n", runtime);
    /* (iterations) * (8 flops / register) * (16 registers / iteration) */
    printf("gflops: %.12f\n", N * 8 * 16 / (runtime * 1e9));

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
