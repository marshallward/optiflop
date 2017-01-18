/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <time.h>       /* timespec, clock_gettime */
#include "gettime.h"

const double TEST_ADD_ADD = 1.4142135623730950488;
const double TEST_ADD_SUB = 1.414213562373095;

const uint64_t N = 1000000000;

#define USE_RDTSC
//const double CPUFREQ = 2.593966925e9;     // My desktop?
const double CPUFREQ = 2.601e9;             // Raijin (?)

/* Headers */
float reduce_AVX(__m256);

int main(int argc, char *argv[])
{
    #pragma omp parallel
    {
        __m256 r[4];

        const __m256 add0 = _mm256_set1_ps((float)TEST_ADD_ADD);
        const __m256 sub0 = _mm256_set1_ps((float)TEST_ADD_SUB);

        uint64_t i, j;
        float result;

        double runtime;

#ifdef USE_RDTSC
        /* TODO: Make some sort of hash table */
        void (*get_starttime) (struct RDTSC_TimeContext *);
        void (*get_endtime) (struct RDTSC_TimeContext *);
        double (*get_runtime) (struct RDTSC_TimeContext *);
        struct RDTSC_TimeContext *time;

        printf("starting rdtsc...\n");

        time = malloc(sizeof(struct RDTSC_TimeContext));
        time->cpufreq = CPUFREQ;
        get_starttime = &rdtsc_starttime;
        get_endtime = &rdtsc_endtime;
        get_runtime = &rdtsc_runtime;
#else
        void (*get_starttime) (struct POSIX_TimeContext *);
        void (*get_endtime) (struct POSIX_TimeContext *);
        double (*get_runtime) (struct POSIX_TimeContext *);
        struct POSIX_TimeContext *time;

        printf("starting posix...\n");

        time = malloc(sizeof(struct POSIX_TimeContext));
        time->clock = CLOCK_MONOTONIC_RAW;
        get_starttime = &posix_starttime;
        get_endtime = &posix_endtime;
        get_runtime = &posix_runtime;
#endif

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

        (*get_starttime)(time);
        for (i = 0; i < N; i++) {
            for (j = 0; j < 4; j++)
                r[j] = _mm256_add_ps(r[j], add0);

            for (j = 0; j < 4; j++)
                r[j] = _mm256_sub_ps(r[j], sub0);
        }
        (*get_endtime)(time);
        runtime = (*get_runtime)(time);

        /* In order to prevent removal of the prior loop by optimisers,
         * sum the register values and print the result. */

        /* Binomial reduction sum */
        r[0] = _mm256_add_ps(r[0], r[2]);
        r[1] = _mm256_add_ps(r[1], r[3]);
        r[0] = _mm256_add_ps(r[0], r[1]);

        /* Sum of AVX registers */
        result = reduce_AVX(r[0]);

        printf("result: %f\n", result);
        printf("runtime: %.12f\n", runtime);
        /* (iterations) * (8 flops / register) * (8 registers / iteration) */
        printf("gflops: %.12f\n", N * 8 * 8 / (runtime * 1e9));
    }

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
